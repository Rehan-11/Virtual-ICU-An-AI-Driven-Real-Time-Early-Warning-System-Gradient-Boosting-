# run_icu.py
"""
Virtual ICU Monitor Launcher
Run this script to start the Streamlit application
"""
import subprocess
import sys
import os

def main():
    """Launch the Virtual ICU Monitor"""
    print("🏥 Starting Virtual ICU Monitor...")
    print("📊 Loading patient data and initializing dashboard...")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/ui_app.py",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Virtual ICU Monitor stopped.")

if __name__ == "__main__":
    main()
