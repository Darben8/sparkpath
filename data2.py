import json
import pandas as pd

# Load JSON file
with open("entertainment.json", "r") as f:
    data = json.load(f)

# Pre-defined RIASEC scores for each role (approximate, based on typical personality fit)
riasec_scores = {
    "Video Editor": {"R":2,"I":1,"A":5,"S":1,"E":1,"C":2},
    "Cinematographer": {"R":2,"I":2,"A":5,"S":1,"E":2,"C":2},
    "Camera Operator": {"R":3,"I":1,"A":4,"S":1,"E":1,"C":2},
    "Film Director": {"R":2,"I":2,"A":5,"S":1,"E":5,"C":2},
    "Producer": {"R":2,"I":2,"A":3,"S":2,"E":5,"C":4},
    "Writer / Author": {"R":1,"I":3,"A":5,"S":2,"E":1,"C":2},
    "Sound Engineering Technician": {"R":4,"I":3,"A":3,"S":1,"E":1,"C":3},
    "Audio/Video Technician": {"R":4,"I":2,"A":3,"S":1,"E":1,"C":4},
    "Makeup Artist (Theatrical)": {"R":1,"I":1,"A":5,"S":2,"E":1,"C":2},
    "Set & Exhibit Designer": {"R":3,"I":2,"A":5,"S":1,"E":2,"C":3},
    "News Analyst / Reporter": {"R":1,"I":4,"A":2,"S":3,"E":2,"C":3},
    "Entertainer & Performer (Other)": {"R":2,"I":1,"A":5,"S":3,"E":3,"C":1},
    "Actor": {"R":2,"I":1,"A":5,"S":3,"E":3,"C":1},
    "Dancer": {"R":2,"I":1,"A":5,"S":3,"E":2,"C":1},
    "Choreographer": {"R":2,"I":1,"A":5,"S":2,"E":3,"C":2},
    "Animator": {"R":2,"I":2,"A":5,"S":1,"E":2,"C":3},
    "Graphic Designer": {"R":1,"I":2,"A":5,"S":1,"E":2,"C":3},
    "Content Creator": {"R":2,"I":2,"A":5,"S":2,"E":3,"C":2},
    "Musician / Singer": {"R":2,"I":1,"A":5,"S":3,"E":2,"C":1},
    "Music Director / Composer": {"R":2,"I":2,"A":5,"S":2,"E":3,"C":3},
    "Radio & TV Announcer": {"R":1,"I":2,"A":3,"S":3,"E":3,"C":2},
    "Marketing Specialist / Publicist": {"R":1,"I":3,"A":3,"S":3,"E":5,"C":3},
    "Business Operations Specialist": {"R":2,"I":3,"A":2,"S":2,"E":4,"C":5},
    "Event Planner": {"R":1,"I":2,"A":3,"S":3,"E":4,"C":4},
    "General & Operations Manager": {"R":2,"I":3,"A":2,"S":3,"E":5,"C":5},
    "HR Specialist (Talent Recruitment)": {"R":1,"I":3,"A":2,"S":5,"E":4,"C":4},
    "Lawyer": {"R":2,"I":5,"A":1,"S":2,"E":4,"C":5},
    "Accountant": {"R":3,"I":4,"A":1,"S":2,"E":2,"C":5},
    "Office Clerk": {"R":2,"I":2,"A":1,"S":2,"E":2,"C":4},
    "Talent Agent": {"R":1,"I":2,"A":2,"S":3,"E":5,"C":3},
}

# Flatten JSON and add RIASEC scores
rows = []
for role in data:
    scores = riasec_scores.get(role["title"], {"R":0,"I":0,"A":0,"S":0,"E":0,"C":0})
    rows.append({
        "Title": role["title"],
        "SOC_Code": role["soc_code"],
        "Cluster": role["cluster"],
        "Description": role["description"],
        "Skills": "; ".join(role.get("skills", [])),
        "Tools": "; ".join(role.get("tools", [])),
        "Education_Level": role.get("education_level",""),
        "Experience_Needed": role.get("experience_needed",""),
        "Training": role.get("training",""),
        "Certifications": "; ".join(role.get("certifications", [])),
        "Salary_Range": role.get("salary_range",""),
        "Resources": "; ".join(role.get("resources", [])),
        "R": scores["R"],
        "I": scores["I"],
        "A": scores["A"],
        "S": scores["S"],
        "E": scores["E"],
        "C": scores["C"]
    })

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rows)
df.to_csv("entertainment_roles_riasec.csv", index=False)
print("CSV saved as entertainment_roles_riasec.csv")
