import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the required VADER lexicon file from NLTK natively
nltk.download('vader_lexicon', quiet=True)

def generate_multi_source_text_data():
    """Generates a realistic multi-source text dataset mimicking social, retail, and news feeds."""
    raw_data = [
        # --- Amazon Product Reviews ---
        {"Source": "Amazon Reviews", "Text": "This smartphone is absolutely amazing! The camera quality is stellar and battery lasts two days. Highly recommended."},
        {"Source": "Amazon Reviews", "Text": "Terrible build quality. The screen cracked within two hours of normal usage. Waste of money."},
        {"Source": "Amazon Reviews", "Text": "The headphones are okay. Sound profile is decent for the low price point, but nothing spectacular."},
        {"Source": "Amazon Reviews", "Text": "Extremely disappointed. Item arrived broken, and the customer support team refused to issue a refund."},
        {"Source": "Amazon Reviews", "Text": "Best laptop I have ever owned. Super fast processing speeds and smooth display."},
        
        # --- Social Media Platforms ---
        {"Source": "Social Media", "Text": "Wow, the new software update completely broke my user interface. Everything feels sluggish today. #fail"},
        {"Source": "Social Media", "Text": "Huge thanks to the dev team for fixing the login bugs so quickly! Outstanding support! 🚀"},
        {"Source": "Social Media", "Text": "Just scrolling through the feed, eating lunch, and watching the rain outside. Peaceful day."},
        {"Source": "Social Media", "Text": "I hate waiting in long customer service queues. The hold music is absolutely driving me crazy right now."},
        {"Source": "Social Media", "Text": "Met up with friends today and tried out the new cafe downtown. The iced lattes were brilliant!"},
        
        # --- News Headsets & Portals ---
        {"Source": "News Sites", "Text": "Tech stocks plummet globally as manufacturing supply chains face unprecedented delays and shortages."},
        {"Source": "News Sites", "Text": "Local green energy startup secures $50M in funding to expand solar grid infrastructure nationwide."},
        {"Source": "News Sites", "Text": "The municipal council held an ordinary meeting on Tuesday evening to discuss routine neighborhood zoning guidelines."},
        {"Source": "News Sites", "Text": "Medical breakthrough announced as new trial drug shows highly positive results against chronic illnesses."},
        {"Source": "News Sites", "Text": "Economic inflation hits record highs this fiscal quarter, prompting deep concern among retail consumers."}
    ]
    df = pd.DataFrame(raw_data)
    df.to_csv('customer_feedback_feed.csv', index=False)
    print("📁 Success! Raw text corpus file 'customer_feedback_feed.csv' generated locally.")

def analyze_and_map_sentiment():
    # Load dataset
    df = pd.read_csv('customer_feedback_feed.csv')
    
    # Initialize the Lexicon Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    
    # 1. Apply NLP Lexicon scoring rules
    # VADER returns a compound score between -1 (extreme negative) and +1 (extreme positive)
    print("\n" + "="*60)
    print(" PHASE 1: COMPUTING LEXICON POLARITY SCORES (NLP)")
    print("="*60)
    
    df['Compound_Score'] = df['Text'].apply(lambda txt: sia.polarity_scores(txt)['compound'])
    
    # Classification logic based on standardized academic thresholds
    def categorize_score(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
            
    df['Sentiment'] = df['Compound_Score'].apply(categorize_score)
    print("Scores computed. Sample mapping:")
    print(df[['Source', 'Sentiment', 'Compound_Score']].head())

    # 2. Derive strategic Business Insights
    print("\n" + "="*60)
    print(" PHASE 2: EXTRACTING STRATEGIC SOCIAL & PRODUCT INSIGHTS")
    print("="*60)
    
    # Cross-tabulate metrics by source channel
    source_distribution = pd.crosstab(df['Source'], df['Sentiment'])
    print("📊 Sentiment Volume Distribution by Media Channel:")
    print(source_distribution)
    
    # Business actionable strategy derived from patterns
    print("\n💡 Marketing & Product Action Items:")
    print("- Amazon Reviews: Target negative entries instantly to deploy customer retention interventions.")
    print("- Social Media: Capitalize on positive tags (#brilliant) for organic marketing amplification.")
    print("- News Portals: Track structural macro shifts to pivot core product pricing models ahead of market changes.")

    # 3. Structural Advanced Data Visualization
    print("\n" + "="*60)
    print(" PHASE 3: COMPILING STRATEGIC DASHBOARD VISUALIZATIONS")
    print("="*60)
    print("📈 Opening matplotlib window to display analytical text dynamics...")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Cross-Channel Text Sentiment Intelligence Dashboard\nPortfolio Project - Aastha Jain (CodeAlpha Intern)', 
                 fontsize=15, fontweight='bold', color='#1e293b')
    
    # Plot 1: Visualizing structural distribution volume by Source
    sns.countplot(ax=axes[0], data=df, x='Source', hue='Sentiment', palette=['green', 'red', 'gray'])
    axes[0].set_title('Sentiment Volume Patterns Across Media Channels', fontsize=12, fontweight='semibold')
    axes[0].set_xlabel('Data Feed Source', fontsize=11)
    axes[0].set_ylabel('Frequency Count', fontsize=11)
    
    # Plot 2: Boxplot mapping distributions of exact numeric intensity scores
    sns.boxplot(ax=axes[1], data=df, x='Source', y='Compound_Score', hue='Source', palette='Set3', legend=False)
    sns.stripplot(ax=axes[1], data=df, x='Source', y='Compound_Score', color='black', alpha=0.6, size=6)
    axes[1].axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
    axes[1].axhline(-0.05, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
    axes[1].set_title('Polarity Intensity Spread by Source', fontsize=12, fontweight='semibold')
    axes[1].set_xlabel('Data Feed Source', fontsize=11)
    axes[1].set_ylabel('NLP Compound Score Limit Range', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_dashboard.png', dpi=300)
    print("💾 Analysis Dashboard visual saved cleanly as 'sentiment_analysis_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    generate_multi_source_text_data()
    analyze_and_map_sentiment()
