import pandas as pd
import os

# Configuration
REPORTS_DIR = "mule_detection/reports/"

def generate_submission():
    print("Generating submission.csv...")
    alerts = pd.read_csv(os.path.join(REPORTS_DIR, "suspicious_alerts.csv"))
    
    # Selecting required columns
    submission = alerts[['account_id', 'predicted_label', 'risk_score', 'risk_category']]
    
    submission.to_csv("submission.csv", index=False)
    print("submission.csv generated in root folder.")

if __name__ == "__main__":
    generate_submission()
