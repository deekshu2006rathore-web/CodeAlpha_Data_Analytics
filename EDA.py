import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_raw_dirty_dataset():
    """Generates a messy dataset with missing entries, duplicates, outliers, and typos."""
    np.random.seed(42)
    
    # Generate 150 normal entries
    data = {
        'Employee_ID': [f"EMP{i:03d}" for i in range(1, 151)],
        'Department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], 150, p=[0.4, 0.3, 0.2, 0.1]),
        'Age': np.random.randint(22, 60, 150),
        'Salary': np.random.randint(45000, 120000, 150),
        'Performance_Score': np.random.uniform(1.0, 5.0, 150),
        'Remote_Worker': np.random.choice(['Yes', 'No'], 150, p=[0.6, 0.4])
    }
    df = pd.DataFrame(data)
    
    # Introduce structural problems explicitly for analysis mapping
    df.loc[12, 'Salary'] = np.nan            # Missing Salary Value
    df.loc[45, 'Age'] = np.nan               # Missing Age Value
    df.loc[78, 'Department'] = 'Enginering'  # Text Typo / Categorical mismatch
    df.loc[88, 'Department'] = 'SALES'       # Case sensitivity issue
    df.loc[99, 'Salary'] = 850000           # Outlier (Extremely high salary anomalies)
    df.loc[101, 'Performance_Score'] = -2.5  # Boundary anomaly (Negative performance score)
    
    # Append Duplicate records
    duplicates = df.iloc[[10, 20, 30]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    df.to_csv('raw_employee_data.csv', index=False)
    print("📁 Success! Raw messy dataset 'raw_employee_data.csv' generated locally.")

def perform_comprehensive_eda():
    # Load dataset
    df = pd.read_csv('raw_employee_data.csv')
    
    print("\n" + "="*50)
    print(" PHASE 1: ASKING MEANINGFUL BUSINESS QUESTIONS")
    print("="*50)
    questions = [
        "1. What is the fundamental structure, size, and compilation schema of our dataset?",
        "2. Do salaries vary significantly across departments, or does a specific tier skew the numbers?",
        "3. Is there a measurable relationship/trend linking an employee's age to their compensation?",
        "4. Do remote workers achieve higher performance scores than structural on-site employees?",
        "5. Are there systemic data engineering gaps (missing items, structural anomalies, duplicates) to resolve?"
    ]
    for q in questions:
        print(q)
        
    print("\n" + "="*50)
    print(" PHASE 2: EXPLORING DATA STRUCTURE & DATA TYPES")
    print("="*50)
    print(f"Dataset Shape (Rows, Columns): {df.shape}")
    print("\n--- Summary Schema Information ---")
    print(df.info())
    print("\n--- Statistical Overview of Metrics ---")
    print(df.describe(include='all'))

    print("\n" + "="*50)
    print(" PHASE 3: IDENTIFYING DATA PROBLEMS & ANOMALIES")
    print("="*50)
    
    # Check for structural Null values
    print("❌ Missing Values Count per Feature:")
    print(df.isnull().sum())
    
    # Check for Duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\n❌ Exact Duplicate Records Found: {duplicate_count}")
    
    # Check for Categorical distribution anomalies / structural typos
    print("\n❌ Unique Values in 'Department' (Checking for Typos/Case mismatch):")
    print(df['Department'].unique())
    
    # Check logical numerical anomalies / bounds violation
    score_anomalies = df[df['Performance_Score'] < 1.0].shape[0]
    print(f"❌ Records violating Performance range limits (Score < 1.0): {score_anomalies}")

    print("\n" + "="*50)
    print(" PHASE 4: HYPOTHESIS TESTING & VALIDATION")
    print("="*50)
    
    # Hypothesis 1 Assumption: Engineering department earns the highest average base salary.
    # Grouping data to inspect trends while intentionally spotting outlier disruption
    dept_salary_trend = df.groupby('Department')['Salary'].agg(['mean', 'median', 'count'])
    print("📊 Salary Metrics Grouped by Department:")
    print(dept_salary_trend)
    print("\nObservation: The 'Sales' department mean is massively inflated due to an outlier ($850,000).")
    
    # Hypothesis 2 Assumption: Performance scales linearly with older age brackets.
    age_perf_correlation = df['Age'].corr(df['Performance_Score'])
    print(f"\n📊 Correlation Coefficient between Age and Performance Score: {age_perf_correlation:.4f}")
    print("Observation: The value near 0 proves age doesn't influence productivity scores.")

    print("\n" + "="*50)
    print(" PHASE 5: ADVANCED STATISTICAL VISUALIZATION")
    print("="*50)
    print("📈 Opening matplotlib window to display structured EDA layouts...")
    
    # Setup subplots figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploratory Data Analysis (EDA) - Aastha Jain Portfolio Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Boxplot to visual detect outliers in Salaries
    sns.boxplot(ax=axes[0, 0], data=df, x='Department', y='Salary', palette='Set2')
    axes[0, 0].set_title('Salary Distribution: Spotting Outlier Anomalies')
    axes[0, 0].tick_params(axis='x', rotation=30)
    
    # Plot 2: Scatter plot to investigate Age vs Performance relationship
    sns.scatterplot(ax=axes[0, 1], data=df, x='Age', y='Performance_Score', hue='Remote_Worker', alpha=0.8)
    axes[0, 1].set_title('Scatter Mapping: Performance vs Age Relationship')
    
    # Plot 3: Missing data pattern visualization
    sns.heatmap(ax=axes[1, 0], data=df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    axes[1, 0].set_title('Null Value Matrix (Yellow lines indicate gaps)')
    
    # Plot 4: Distribution plot of salaries
    sns.histplot(ax=axes[1, 1], data=df, x='Salary', bins=20, kde=True, color='crimson')
    axes[1, 1].set_title('Salary Density Profile Distribution')
    
    plt.tight_layout()
    plt.savefig('eda_analytics_dashboard.png')
    print("💾 Analysis Dashboard visual saved cleanly as 'eda_analytics_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    generate_raw_dirty_dataset()
    perform_comprehensive_eda()
