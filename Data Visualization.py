import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_portfolio_sales_data():
    """Generates a realistic multi-region raw sales dataset."""
    np.random.seed(101)
    n_records = 400
    
    data = {
        'Date': pd.date_range(start='2026-01-01', periods=n_records, freq='D'),
        'Region': np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], n_records, p=[0.35, 0.25, 0.25, 0.15]),
        'Product_Category': np.random.choice(['Electronics', 'Cloud Software', 'Hardware Platforms', 'AI Accessories'], n_records),
        'Units_Sold': np.random.randint(10, 150, n_records),
        'Unit_Price': np.random.choice([49.99, 199.99, 499.00, 1200.00], n_records, p=[0.4, 0.3, 0.2, 0.1]),
        'Customer_Satisfaction': np.random.uniform(2.5, 5.0, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Revenue'] = df['Units_Sold'] * df['Unit_Price']
    df['Month'] = df['Date'].dt.strftime('%B')
    
    # Introduce structural seasonal growth to create a real story
    df.loc[df['Month'].isin(['October', 'November', 'December']), 'Revenue'] *= 1.4
    
    df.to_csv('portfolio_sales_data.csv', index=False)
    print("📁 'portfolio_sales_data.csv' successfully created for visualization processing.")

def create_impactful_dashboard():
    # Load dataset
    df = pd.read_csv('portfolio_sales_data.csv')
    
    # Set professional style environment
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Initialize a 2x2 structural canvas
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Global Executive Sales Analytics Dashboard\nPortfolio Project - Aastha Jain (CodeAlpha Intern)', 
                 fontsize=18, fontweight='bold', color='#1a252c', y=0.98)
    
    # ------------------------------------------------------------------
    # QUADRANT 1: Bar Chart - Total Revenue by Product Category (Structural Insight)
    # ------------------------------------------------------------------
    cat_revenue = df.groupby('Product_Category')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
    sns.barplot(ax=axes[0, 0], data=cat_revenue, x='Revenue', y='Product_Category', palette='viridis', hue='Product_Category', legend=False)
    axes[0, 0].set_title('Revenue Breakdown by Product Line', fontsize=13, fontweight='semibold', pad=10)
    axes[0, 0].set_xlabel('Total Revenue (USD $)', fontsize=11)
    axes[0, 0].set_ylabel('', fontsize=11)
    # Format labels natively
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}K".format(int(x/1000))))

    # ------------------------------------------------------------------
    # QUADRANT 2: Line Graph - Revenue Trend Over Time (Seasonal Insight)
    # ------------------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_trend = df.resample('M', on='Date')['Revenue'].sum().reset_index()
    axes[0, 1].plot(monthly_trend['Date'], monthly_trend['Revenue'], marker='o', linewidth=2.5, color='#d9534f', label='Monthly Revenue')
    axes[0, 1].fill_between(monthly_trend['Date'], monthly_trend['Revenue'], color='#d9534f', alpha=0.15)
    axes[0, 1].set_title('Macro Revenue Growth Timeline (Linear Run-Rate)', fontsize=13, fontweight='semibold', pad=10)
    axes[0, 1].set_xlabel('Timeline Horizon', fontsize=11)
    axes[0, 1].set_ylabel('Revenue ($)', fontsize=11)
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}K".format(int(x/1000))))

    # ------------------------------------------------------------------
    # QUADRANT 3: Donut/Pie Chart - Market Share Distribution by Region
    # ------------------------------------------------------------------
    region_share = df.groupby('Region')['Revenue'].sum()
    colors = ['#2b5c8f', '#4682b4', '#6baed6', '#b3cde3']
    wedges, texts, autotexts = axes[1, 0].pie(region_share, labels=region_share.index, autopct='%1.1f%%', 
                                  startangle=140, colors=colors, textprops=dict(color="#1a252c"),
                                  wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.setp(autotexts, size=10, weight="bold")
    axes[1, 0].set_title('Global Market Volume Share by Region', fontsize=13, fontweight='semibold', pad=10)

    # ------------------------------------------------------------------
    # QUADRANT 4: Scatter Plot - Units Sold vs. Revenue mapped with Satisfaction
    # ------------------------------------------------------------------
    sns.scatterplot(ax=axes[1, 1], data=df, x='Units_Sold', y='Revenue', 
                    hue='Customer_Satisfaction', palette='coolwarm', size='Units_Sold', sizes=(20, 200), alpha=0.75)
    axes[1, 1].set_title('Customer Satisfaction Density & Pricing Correlation', fontsize=13, fontweight='semibold', pad=10)
    axes[1, 1].set_xlabel('Volume of Units Sold', fontsize=11)
    axes[1, 1].set_ylabel('Gross Order Revenue ($)', fontsize=11)
    axes[1, 1].legend(title='CSAT Score', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Global structural adjustment and fine-tuning spacing
    plt.tight_layout()
    
    # Save the output visualization locally for the repository
    plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
    print("💾 Success! Dashboard visual compiled perfectly and saved as 'sales_dashboard.png'.")
    plt.show()

if __name__ == "__main__":
    generate_portfolio_sales_data()
    create_impactful_dashboard()
