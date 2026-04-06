"""
cron_risk_report.py
--------------------
Run via system crontab 

Add to crontab (crontab -e):
    0 10 * * * /usr/bin/python3 /path/to/cron_risk_report.py >> /var/log/spog_cron.log 2>&1
"""

import json
import re
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from agent2 import (
    registry_loader,
    load_user_credentials,
    _authenticated_client,
    call_llama,
)
from dotenv import load_dotenv
load_dotenv()
import os


CRON_REPORT_RECIPIENTS = os.environ.get("CRON_RECIPIENTS", "").split(",")
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER     = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")



# SEND EMAIL

def send_email_direct(to_email: str, subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to_email, msg.as_string())

    print(f"CRON: Email sent directly to {to_email}")



# Fetch all risk IDs + display_id from risks API

def fetch_all_risk_ids(client, base_url: str) -> list:
    config = registry_loader.get_api_config("risks")
    if not config:
        print("CRON: 'risks' config not found.")
        return []

    url          = f"{base_url}{config.get('endpoint', '')}"
    filter_param = config.get("filter_param", "filters") 
    all_risks    = []
    page         = 1

   
    default_filters = config.get("default_filters", {})  # → {"check_state": ["ready"]}

    while True:
        try:
            params = {
                filter_param: json.dumps(default_filters),  # ← passes check_state=["ready"]
                "page":  page,
                "limit": 100,
            }
            r    = client.get(url, params=params)
            data = r.json().get("data", []) if r.status_code == 200 else []
            if not data:
                break

            for item in data:
                risk_id    = item.get("_id")
                display_id = item.get("check_id") or item.get("display_id") or risk_id
                if risk_id:
                    all_risks.append({
                        "risk_id":    str(risk_id),
                        "display_id": str(display_id),
                        "name":       item.get("name") or item.get("search") or str(display_id),
                    })

            if len(data) < 100:
                break
            page += 1

        except Exception as e:
            print(f"CRON: Error on risks page {page}: {e}")
            break

    print(f"CRON: Collected {len(all_risks)} risk IDs.")
    return all_risks


# ─────────────────────────────────────────────────────────────
# STEP 2: Fetch risk_history for one risk_id
# ─────────────────────────────────────────────────────────────
def fetch_risk_history(client, base_url: str, risk_id: str) -> list:
    config = registry_loader.get_api_config("risk_history")
    if not config:
        return []

    url = f"{base_url}{config.get('endpoint', '')}"
    try:
        r    = client.get(url, params={"risk_id": risk_id, "limit": 100})
        raw  = r.json() if r.status_code == 200 else {}
        data = raw.get("data", raw) if isinstance(raw, dict) else raw

        if not data:
            return []

        records = []
        if isinstance(data, dict):
            for ts, stats in data.items():
                record = {"timestamp": int(ts)}
                record.update(stats)
                records.append(record)
        else:
            records = [{"timestamp": int(x.get("timestamp", 0)), **x} for x in data]

        return sorted(records, key=lambda x: x["timestamp"])

    except Exception as e:
        print(f"CRON: Failed history fetch for risk_id {risk_id}: {e}")
        return []



# STEP 3: Analyze using LLM — same prompt as chat formatter
# Returns { is_consistent, llm_analysis, summary, date_gaps,
#          pass_fail_jumps, remediated_issues, suggestion, latest }

def analyze_with_llm(risk_id: str, display_id: str, name: str, records: list) -> dict:
    if not records:
        return {
            "is_consistent":     False,
            "llm_analysis":      "No history records found for this risk.",
            "summary":           "No data",
            "latest":            "—",
            "date_gaps":         "No data",
            "pass_fail_jumps":   "No data",
            "remediated_issues": "No data",
            "suggestion":        "—",
        }

    readable = []
    for r in records:
        ts   = int(r.get("timestamp", 0))
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d/%b/%y")
        readable.append({
            "date":       date,
            "pass":       r.get("pass", 0),
            "fail":       r.get("fail", 0),
            "remediated": r.get("remediated", 0),
            "total":      r.get("total", 0),
        })

    records_str   = json.dumps(readable, indent=2)
    user_question = f"Analyze risk history for {display_id} ({name})"

    # ── EXACT same prompt as formatter_node ──────────────────
    risk_history_prompt = f"""You are Spog.ai, a smart security analyst assistant.

The user asked: "{user_question}"

Here is the historical risk data sorted oldest to newest:
{records_str}

YOUR JOB:
First, flag the conclusion with coloured statement if the data contains problem with coloured statement red for problem and green for correct, then Write a short, natural analyst briefing. Keep it only to the bullet points, CONCISE, SHORT and to the point, with bullet points only. SHORT

STRICT RULES:
1. Data contains problem if there is large gaps between dates, passed, failed and remediated. incosistency should not happen.
2. START WITH THE CONCLUSION then, Briefly summarize the data — how many snapshots, what date range. (keep it only to the point)
3. Show the latest numbers (pass, fail, remediated).(keep it only to the point)
4. Look at the dates between consecutive records. If there is a large gap between any two dates, flag it — explain that monitoring data may be missing for that period.(keep it only to the point)
5. Look at the pass/fail numbers across records. If there is a major jump or drop between any two consecutive records, flag it and mention the dates.(keep it only to the point)
6. Look at the remediated numbers across records. If there is a major jump or drop between any two consecutive records, flag it and mention the dates.(keep it only to the point)
7. Only if everything looks consistent, say so.
8. End with one helpful follow-up suggestion. (keep it only to the point)

OUTPUT FORMAT — follow this structure exactly:
<span style="color:red">🔴 INCONSISTENCIES DETECTED</span>  (or <span style="color:green">🟢 ALL CLEAR</span> if no problems)

- **Summary:** [rule 2]

- **Latest:** [rule 3]

- **Date gaps:** [rule 4]

- **Pass/Fail jumps:** [rule 5]

- **Remediated records:** [rule 6]

- **Suggestion:** [rule 7]

FORMATTING RULES:
- Each bullet on its OWN line with a BLANK LINE between bullets
- NEVER place two bullets on the same line
- ONLY use `-` dash bullets, NEVER `•`
- Bold all key numbers and dates
- Max 20 words per bullet
"""

    response = call_llama([
        {"role": "system", "content": risk_history_prompt},
        {"role": "user",   "content": "Please give me your briefing now."},
    ])

    # Detect consistency from first line
    first_line    = response.strip().split("\n")[0].lower()
    is_consistent = "all clear" in first_line or "color:green" in first_line

    # Parse each bullet into table columns
    def extract(label: str) -> str:
        pattern = rf'-\s*\*\*{label}[:\*].*?\*?\*?:?\s*(.+?)(?=\n-\s*\*\*|\Z)'
        match   = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip().replace("\n", " ")
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
            return text
        return "—"

    return {
        "is_consistent":     is_consistent,
        "llm_analysis":      response.strip(),
        "summary":           extract("Summary"),
        "latest":            extract("Latest"),
        "date_gaps":         extract("Date gaps"),
        "pass_fail_jumps":   extract("Pass/Fail jumps"),
        "remediated_issues": extract("Remediated records"),
        "suggestion":        extract("Suggestion"),
    }


# ─────────────────────────────────────────────────────────────
# STEP 4: Build HTML email — tabular format
# ─────────────────────────────────────────────────────────────
def build_email(results: list) -> str:
    today        = datetime.now().strftime("%d %b %Y")
    inconsistent = [r for r in results if not r["analysis"]["is_consistent"]]
    consistent   = [r for r in results if     r["analysis"]["is_consistent"]]

    def table_rows(items: list, row_color: str, status_color: str, status_label: str) -> str:
        if not items:
            return f'<tr><td colspan="8" style="padding:12px;text-align:center;color:#999;">None</td></tr>'

        rows = []
        for r in items:
            a = r["analysis"]
            rows.append(f"""
            <tr style="background:{row_color};">
                <td style="padding:10px;border:1px solid #ddd;font-weight:bold;white-space:nowrap;">{r['display_id']}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;max-width:180px;">{r['name']}</td>
                <td style="padding:10px;border:1px solid #ddd;color:{status_color};font-weight:bold;white-space:nowrap;">{status_label}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;">{a['summary']}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;">{a['latest']}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;color:#cc0000;">{a['date_gaps']}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;color:#cc0000;">{a['pass_fail_jumps']}</td>
                <td style="padding:10px;border:1px solid #ddd;font-size:12px;">{a['suggestion']}</td>
            </tr>""")

        return "".join(rows)

    header = """
    <tr style="background:#f0f0f0;">
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Display ID</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Name</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Status</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Summary</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Latest Numbers</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Date Gaps</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Pass/Fail Jumps</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Suggestion</th>
    </tr>"""

    inc_rows = table_rows(inconsistent, "#fff5f5", "#cc0000", "🔴 INCONSISTENT")
    con_rows = table_rows(consistent,   "#f5fff5", "#007700", "🟢 CONSISTENT")

    return f"""<html>
<body style="font-family:Arial,sans-serif;padding:24px;color:#333;">

<h2 style="color:#333;border-bottom:2px solid #eee;padding-bottom:12px;">
    📊 Daily Risk History Report — {today}
</h2>

<p style="font-size:14px;color:#555;margin-bottom:24px;">
    Total risks analyzed: <strong>{len(results)}</strong> &nbsp;|&nbsp;
    <span style="color:#cc0000;">Inconsistent: <strong>{len(inconsistent)}</strong></span> &nbsp;|&nbsp;
    <span style="color:#007700;">Consistent: <strong>{len(consistent)}</strong></span>
</p>

<h3 style="color:#cc0000;margin-top:28px;">⚠️ Inconsistencies Detected ({len(inconsistent)})</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
    {header}
    <tbody>{inc_rows}</tbody>
</table>

<h3 style="color:#007700;margin-top:36px;">✅ All Clear ({len(consistent)})</h3>
<table style="border-collapse:collapse;width:100%;font-size:13px;">
    {header}
    <tbody>{con_rows}</tbody>
</table>

<p style="margin-top:36px;color:#aaa;font-size:11px;border-top:1px solid #eee;padding-top:12px;">
    Sent automatically by Spog.ai Scheduler · {today}
</p>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# MAIN — called directly by crontab
# ─────────────────────────────────────────────────────────────
def run():
    print(f"\n{'='*60}")
    print(f"CRON: daily_risk_history_report started at {datetime.now()}")
    print(f"{'='*60}")

    for recipient_email in CRON_REPORT_RECIPIENTS:
        creds = load_user_credentials(recipient_email)
        if not creds:
            print(f"CRON: No credentials for {recipient_email}, skipping.")
            continue

        base_url = creds.get("api_base_url", "")
        results  = []

        with _authenticated_client(recipient_email) as client:

            # Step 1: collect all risk IDs + display_ids
            all_risks = fetch_all_risk_ids(client, base_url)
            if not all_risks:
                print(f"CRON: No risks found for {recipient_email}, skipping.")
                continue

            # Step 2 + 3: fetch history and LLM analyze each one
            for risk in all_risks:
                print(f"CRON: Processing {risk['display_id']} / {risk['risk_id']} ({risk['name']})...")

                history  = fetch_risk_history(client, base_url, risk["risk_id"])
                analysis = analyze_with_llm(
                    risk["risk_id"],
                    risk["display_id"],
                    risk["name"],
                    history
                )

                results.append({
                    "risk_id":    risk["risk_id"],
                    "display_id": risk["display_id"],
                    "name":       risk["name"],
                    "analysis":   analysis,
                })

                status = "INCONSISTENT ⚠️" if not analysis["is_consistent"] else "CONSISTENT ✅"
                print(f"CRON: {risk['display_id']} → {status}")

        # Step 4: build email
        html    = build_email(results)
        subject = f"📊 Daily Risk History Report — {datetime.now().strftime('%d %b %Y')}"

        
        try:
            send_email_direct(recipient_email, subject, html)
            print(f"CRON: Report sent to {recipient_email} ({len(results)} risks processed)")
        except Exception as e:
            print(f"CRON: Failed to send email to {recipient_email}: {e}")     

    print(f"CRON: Finished at {datetime.now()}\n")


if __name__ == "__main__":
    run()
