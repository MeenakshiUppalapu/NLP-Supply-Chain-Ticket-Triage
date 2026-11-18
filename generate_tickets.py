import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)
n_rows = 5000

carriers = ["CN Rail", "Canada Post", "Challenger Motor Freight", "FedEx Express"]
hubs = [
    "Toronto Hub",
    "Mississauga Hub",
    "Montreal Hub",
    "Vancouver Hub",
    "Chicago Cross-Border Hub",
]

# Define 4 realistic operational ticket categories & templates
ticket_templates = {
    "SLA Delay & Tracking": [
        "Urgent! Shipment {shp} via {carrier} has been sitting at {hub} for over 24 hours with no scan updates. Client SLA is at risk.",
        "Where is our freight {shp}? {carrier} missed the scheduled delivery window out of {hub}. Please advise immediately.",
        "Customer is escalating order {shp}. Tracking shows {carrier} delayed in transit near {hub} due to weather backlog.",
        "Can you check status on {shp}? Need an updated ETA from {carrier} before end of day.",
    ],
    "Damaged / Lost Freight": [
        "CRITICAL: Receiver reported crushed pallets upon delivery of {shp} by {carrier} from {hub}. Filing a cargo claim.",
        "Two cartons are missing from shipment {shp}. {carrier} delivered a partial load at {hub} today.",
        "Freight arrived water-damaged on order {shp}. Need immediate inspection photos and {carrier} claim reference.",
        "Lost package alert for {shp}. {carrier} hub scan at {hub} shows empty container.",
    ],
    "Customs & Cross-Border Hold": [
        "Shipment {shp} is stuck in CBSA customs clearance at {hub}. {carrier} says commercial invoice is missing.",
        "Urgent border hold on {shp} moving via {carrier}. Broker needs updated tariff code to release from {hub}.",
        "Cross-border manifest discrepancy for {shp} with {carrier}. Freight cannot cross until customs paperwork is verified.",
        "Please send USMCA certificate of origin for {shp} currently held at {hub} by {carrier}.",
    ],
    "Billing & Invoice Dispute": [
        "We were overcharged for fuel surcharge on invoice for shipment {shp} moved by {carrier}. Please review.",
        "Duplicate billing detected for {shp} out of {hub}. {carrier} billed our account twice for the same lane.",
        "Requesting credit memo for missed SLA penalty on {shp}. {carrier} failed contracted transit time.",
        "Please clarify detention charges listed on {carrier} statement for shipment {shp} at {hub}.",
    ],
}

urgency_keywords = ["Urgent!", "CRITICAL:", "escalating", "missed", "stuck", "overcharged", "Lost"]

records = []
for i in range(n_rows):
    category = random.choice(list(ticket_templates.keys()))
    template = random.choice(ticket_templates[category])
    shp_id = f"SHP-{random.randint(100000, 109999)}"
    carrier = random.choice(carriers)
    hub = random.choice(hubs)

    # Build raw unstructured email body
    ticket_text = template.format(shp=shp_id, carrier=carrier, hub=hub)

    # Determine ground-truth Urgency Level based on content & category
    if any(word.lower() in ticket_text.lower() for word in ["urgent", "critical", "escalating", "lost", "cbsa"]):
        urgency = "High"
    elif category in ["SLA Delay & Tracking", "Damaged / Lost Freight"]:
        urgency = "Medium"
    else:
        urgency = "Low"

    records.append({
        "ticket_id": f"TKT-{20000 + i}",
        "raw_email_text": ticket_text,
        "department_category": category,
        "urgency_level": urgency,
        "true_shipment_id": shp_id,
        "true_carrier": carrier,
    })

df = pd.DataFrame(records)
df.to_csv("logistics_support_tickets.csv", index=False)

print("NLP Ticket Dataset created successfully! Shape:", df.shape)
print("\nTicket Distribution by Department:")
print(df["department_category"].value_counts())
print("\nTicket Distribution by Urgency:")
print(df["urgency_level"].value_counts())
print("\nSample Ticket:")
print(df.iloc[0].to_dict())