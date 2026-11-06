"""Fictional candidates with a clean, human-written tailored CV for each. No real people.

`DEV` was used to build the checks. `HOLDOUT` was kept aside and scored once at the
end, so its numbers aren't flattered by tuning.
"""
from __future__ import annotations


def persona(name, roles, skills, job_ad, summary, matched, missing, bullets, out_skills, traps):
    parsed = {
        "name": name,
        "contact": "Berlin",
        "roles": [{"title": t, "org": o, "dates": d, "questions": [f"Q{i}" for i in range(len(f))]} for t, o, d, f in roles],
        "skills": skills,
        "education": [],
    }
    answers = {f"r{ri}q{qi}": {"text": fact, "confirmed": True} for ri, r in enumerate(roles) for qi, fact in enumerate(r[3])}
    cv = "\n".join(f"{t}, {o} ({d})" for t, o, d, _ in roles) + "\nSkills: " + ", ".join(skills)
    grouped: dict[int, list] = {}
    for role, text, facts in bullets:
        grouped.setdefault(role, []).append({"text": text, "facts": facts})
    output = {
        "jobTitle": "Role",
        "summary": summary,
        "matched": matched,
        "missing": missing,
        "roles": [{"role": r, "bullets": b} for r, b in grouped.items()],
        "skills": out_skills,
    }
    return {"id": name, "parsed": parsed, "answers": answers, "cv_text": cv, "job_ad": job_ad, "clean": output, "traps": traps}


DEV = [
    persona(
        "Mira Ops",
        [("Operations Analyst", "Example Bank", "2022-2025", [
            "Reconciled partner bank files every morning using Excel and SQL",
            "Reduced monthly report errors by 15% after adding control checks",
            "Wrote process documentation for onboarding new partner banks",
        ]), ("Customer Support Agent", "Example Telecom", "2020-2022", [
            "Answered billing questions for about 60 customers a day",
            "Logged recurring problems in Jira",
        ])],
        ["Excel", "SQL", "Jira"],
        "Business Analyst, fintech. Requirements: SQL, Tableau, stakeholder management, process documentation, reconciliation, Python.",
        "Operations analyst with experience in partner bank reconciliation and control checks. Uses Excel, SQL and Jira.",
        ["SQL", "process documentation", "reconciliation"],
        ["Tableau", "Python"],
        [(0, "Reconciled partner bank files every morning using SQL and Excel", ["r0q0"]),
         (0, "Added control checks that reduced monthly report errors by 15%", ["r0q1"]),
         (0, "Wrote process documentation for partner bank onboarding", ["r0q2"]),
         (1, "Logged recurring customer problems in Jira", ["r1q1"])],
        ["SQL", "Excel", "Jira"],
        {"ad_tool": "Tableau", "other_tool": "Kafka", "ad_skill": "Python", "supported": "Excel",
         "vague": ", earning praise from senior leadership", "inflate": "Led"},
    ),
    persona(
        "Jonas Warehouse",
        [("Warehouse Associate", "Example Logistics", "2021-2025", [
            "Picked and packed around 200 orders per shift",
            "Trained 3 new colleagues on the handheld scanner",
            "Did stock counts every Friday",
        ]), ("Delivery Assistant", "Example Foods", "2019-2021", [
            "Loaded vans and checked delivery notes",
        ])],
        ["Handheld scanners", "Stock counts"],
        "Shift Lead, warehouse. Requirements: forklift licence, SAP, team leadership, stock counts, shift planning.",
        "Warehouse associate experienced in order picking and weekly stock counts, who has trained new colleagues.",
        ["stock counts"],
        ["forklift licence", "SAP", "shift planning"],
        [(0, "Picked and packed around 200 orders per shift", ["r0q0"]),
         (0, "Trained 3 new colleagues on the handheld scanner", ["r0q1"]),
         (0, "Ran stock counts every Friday", ["r0q2"]),
         (1, "Loaded vans and checked delivery notes", ["r1q0"])],
        ["Stock counts", "Handheld scanners"],
        {"ad_tool": "SAP", "other_tool": "Oracle", "ad_skill": "Forklift licence", "supported": "stock counts",
         "vague": ", consistently exceeding expectations", "inflate": "Managed"},
    ),
    persona(
        "Sara Marketing",
        [("Marketing Executive", "Example Shop", "2023-2025", [
            "Ran weekly email campaigns in Mailchimp",
            "Grew the newsletter list from 4,000 to 9,500 subscribers",
        ]), ("Marketing Intern", "Example Agency", "2022-2023", [
            "Wrote social media posts for 5 clients",
            "Pulled campaign reports from Google Analytics",
        ])],
        ["Mailchimp", "Google Analytics", "Copywriting"],
        "Growth Marketer. Requirements: HubSpot, paid social, A/B testing, Google Analytics, email campaigns.",
        "Marketing executive who grew a newsletter list from 4,000 to 9,500 subscribers through weekly email campaigns.",
        ["Google Analytics", "email campaigns"],
        ["HubSpot", "paid social", "A/B testing"],
        [(0, "Ran weekly email campaigns in Mailchimp", ["r0q0"]),
         (0, "Grew newsletter list from 4,000 to 9,500 subscribers", ["r0q1"]),
         (1, "Pulled campaign reports from Google Analytics", ["r1q1"]),
         (1, "Wrote social media posts for 5 clients", ["r1q0"])],
        ["Google Analytics", "Mailchimp", "Copywriting"],
        {"ad_tool": "HubSpot", "other_tool": "Salesforce", "ad_skill": "A/B testing", "supported": "Mailchimp",
         "vague": ", which transformed how the company reached customers", "inflate": "Headed"},
    ),
    persona(
        "Tomasz Dev",
        [("Backend Developer", "Example SaaS", "2021-2025", [
            "Built REST APIs in Python and FastAPI",
            "Cut average API response time from 800 ms to 250 ms",
            "Wrote integration tests with pytest",
        ]), ("Junior Developer", "Example Studio", "2019-2021", [
            "Fixed bugs in a PHP web application",
        ])],
        ["Python", "FastAPI", "pytest", "PHP"],
        "Senior Backend Engineer. Requirements: Python, Kubernetes, AWS, PostgreSQL, API design.",
        "Backend developer who builds REST APIs in Python and FastAPI, and cut API response time from 800 ms to 250 ms.",
        ["Python", "REST APIs"],
        ["Kubernetes", "AWS", "PostgreSQL"],
        [(0, "Built REST APIs in Python and FastAPI", ["r0q0"]),
         (0, "Cut average API response time from 800 ms to 250 ms", ["r0q1"]),
         (0, "Wrote integration tests with pytest", ["r0q2"]),
         (1, "Fixed bugs in a PHP web application", ["r1q0"])],
        ["Python", "FastAPI", "pytest"],
        {"ad_tool": "Kubernetes", "other_tool": "Kafka", "ad_skill": "AWS", "supported": "Python",
         "vague": ", becoming the go-to expert for the whole company", "inflate": "Directed"},
    ),
]

HOLDOUT = [
    persona(
        "Lena Nurse",
        [("Staff Nurse", "Example Clinic", "2020-2025", [
            "Cared for up to 12 patients per shift on a surgical ward",
            "Documented care plans in the Orbis system",
            "Mentored 2 trainee nurses during their placements",
        ]), ("Care Assistant", "Example Care Home", "2018-2020", [
            "Helped residents with daily routines and meals",
        ])],
        ["Orbis", "Wound care", "Patient documentation"],
        "Ward Manager. Requirements: staff rostering, budget responsibility, Orbis, wound care, quality audits.",
        "Staff nurse on a surgical ward with experience in wound care and patient documentation, who has mentored trainee nurses.",
        ["Orbis", "wound care"],
        ["staff rostering", "budget responsibility", "quality audits"],
        [(0, "Looked after up to 12 patients per shift on a surgical ward", ["r0q0"]),
         (0, "Recorded care plans in Orbis", ["r0q1"]),
         (0, "Mentored 2 trainee nurses through their placements", ["r0q2"]),
         (1, "Assisted residents with daily routines and meals", ["r1q0"])],
        ["Orbis", "Wound care", "Patient documentation"],
        {"ad_tool": "rostering", "other_tool": "Epic", "ad_skill": "Budget responsibility", "supported": "wound care",
         "vague": ", gaining a reputation as the most dependable nurse on the ward", "inflate": "Supervised"},
    ),
    persona(
        "Priya Finance",
        [("Accounts Payable Clerk", "Example Retail", "2021-2025", [
            "Processed about 1,200 supplier invoices a month in DATEV",
            "Matched bank statements to open items at month end",
        ]), ("Finance Intern", "Example Consulting", "2020-2021", [
            "Built an Excel tracker for travel expenses",
            "Prepared data for the quarterly VAT return",
        ])],
        ["DATEV", "Excel", "Bank reconciliation"],
        "Financial Controller. Requirements: IFRS, SAP FI, variance analysis, month-end close, DATEV.",
        "Accounts payable clerk who processes about 1,200 invoices a month in DATEV and matches bank statements at month end.",
        ["DATEV", "month-end"],
        ["IFRS", "SAP FI", "variance analysis"],
        [(0, "Processed about 1,200 supplier invoices monthly in DATEV", ["r0q0"]),
         (0, "Matched bank statements to open items at month end", ["r0q1"]),
         (1, "Built an Excel tracker for travel expenses", ["r1q0"]),
         (1, "Prepared data for the quarterly VAT return", ["r1q1"])],
        ["DATEV", "Excel", "Bank reconciliation"],
        {"ad_tool": "IFRS", "other_tool": "NetSuite", "ad_skill": "Variance analysis", "supported": "Excel",
         "vague": ", saving the business a great deal of money", "inflate": "Oversaw"},
    ),
    persona(
        "Ahmed Hotel",
        [("Front Desk Agent", "Example Hotel", "2022-2025", [
            "Checked in about 80 guests a day using Opera",
            "Handled guest complaints and escalated refunds to the duty manager",
        ]), ("Waiter", "Example Bistro", "2019-2022", [
            "Served tables and took payments at the till",
            "Helped with stock orders on weekends",
        ])],
        ["Opera", "Guest relations", "English", "Arabic"],
        "Revenue Manager, hotel. Requirements: revenue forecasting, Opera, IDeaS, OTA channel management, guest relations.",
        "Front desk agent experienced in guest relations and Opera, who checks in about 80 guests a day.",
        ["Opera", "guest relations"],
        ["revenue forecasting", "IDeaS", "OTA channel management"],
        [(0, "Checked in about 80 guests a day using Opera", ["r0q0"]),
         (0, "Resolved guest complaints and passed refunds to the duty manager", ["r0q1"]),
         (1, "Served tables and took payments at the till", ["r1q0"]),
         (1, "Helped with weekend stock orders", ["r1q1"])],
        ["Opera", "Guest relations", "Arabic"],
        {"ad_tool": "IDeaS", "other_tool": "Amadeus", "ad_skill": "Revenue forecasting", "supported": "Opera",
         "vague": ", delighting guests and lifting the hotel's reputation", "inflate": "Led"},
    ),
    persona(
        "Clara Data",
        [("Data Analyst", "Example Insurance", "2022-2025", [
            "Built Power BI dashboards for the claims team",
            "Wrote SQL queries to find duplicate claims, which flagged 340 duplicates in one year",
            "Presented monthly findings to team leads",
        ]), ("Working Student", "Example University", "2020-2022", [
            "Cleaned survey data in Python",
        ])],
        ["Power BI", "SQL", "Python"],
        "Analytics Engineer. Requirements: dbt, Snowflake, SQL, data modelling, Airflow.",
        "Data analyst who builds Power BI dashboards and used SQL queries to flag 340 duplicate claims in one year.",
        ["SQL"],
        ["dbt", "Snowflake", "data modelling", "Airflow"],
        [(0, "Built Power BI dashboards for the claims team", ["r0q0"]),
         (0, "Used SQL queries to flag 340 duplicate claims in one year", ["r0q1"]),
         (0, "Presented findings to team leads each month", ["r0q2"]),
         (1, "Cleaned survey data in Python", ["r1q0"])],
        ["SQL", "Power BI", "Python"],
        {"ad_tool": "Snowflake", "other_tool": "Looker", "ad_skill": "Airflow", "supported": "SQL",
         "vague": ", influencing the company's whole strategy", "inflate": "Spearheaded"},
    ),
]
