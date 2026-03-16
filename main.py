"""
MAIN PIPELINE – Run All Steps in Order
AI-Based Real Estate Valuation System

Usage:
    python main.py           # Run full pipeline
    python main.py --step 1  # Step 1: Preprocessing only
    python main.py --step 2  # Step 2: EDA only
    python main.py --step 3  # Step 3: Model training only
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def step1():
    print("\n" + "█" * 60)
    print("  STEP 1 — Data Collection, Preprocessing & Feature Engineering")
    print("█" * 60)
    from src.data_preprocessing import build_final_dataset
    build_final_dataset()


def step2():
    print("\n" + "█" * 60)
    print("  STEP 2 — Exploratory Data Analysis (EDA)")
    print("█" * 60)
    from src.eda import run_eda
    run_eda()


def step3():
    print("\n" + "█" * 60)
    print("  STEP 3 — Model Training & Evaluation")
    print("█" * 60)
    from src.model_training import train_all_models
    train_all_models()


def step4():
    print("\n" + "█" * 60)
    print("  STEP 4 — Launching Streamlit App")
    print("█" * 60)
    print("  Run the following command to launch the app:")
    print()
    print("      streamlit run src/app.py")
    print()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "--step":
        step_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        steps = {1: step1, 2: step2, 3: step3, 4: step4}
        if step_num in steps:
            steps[step_num]()
        else:
            print("Invalid step. Use --step 1, 2, 3, or 4")
    else:
        # Full pipeline
        print("\n" + "=" * 60)
        print("  AI-Based Real Estate Valuation System")
        print("  Full Pipeline Starting...")
        print("=" * 60)
        step1()
        step2()
        step3()
        step4()
        print("\n✅ All steps complete! Launch the app with:")
        print("   streamlit run src/app.py\n")
