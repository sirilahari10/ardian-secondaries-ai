import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# DATA ENGINEERING: SIMULATE PE FUND METRICS
def generate_fund_data(num_samples=1000):
    print("[INFO] Ingesting historical PE fund performance data...")
    np.random.seed(42)
    
    vintages = np.random.choice(range(2012, 2024), num_samples)
    strategies = np.random.choice(['Buyout', 'Growth', 'Venture', 'Infrastructure'], num_samples)
    tvpi = np.random.uniform(1.1, 3.5, num_samples)
    dpi = np.random.uniform(0.1, 1.5, num_samples)
    nav_usd = np.random.uniform(5e6, 50e6, num_samples)
    

    base_pricing = 0.85 + (dpi * 0.1) + ((tvpi - 1.5) * 0.05)
    
    actual_pricing_pct = np.clip(base_pricing + np.random.normal(0, 0.03, num_samples), 0.70, 1.10)
    
    df = pd.DataFrame({
        'fund_id': [f"Fund-{i:05d}" for i in range(num_samples)],
        'vintage': vintages,
        'strategy': strategies,
        'tvpi': tvpi,
        'dpi': dpi,
        'nav_usd': nav_usd,
        'actual_pricing_pct': actual_pricing_pct
    })
    
    df = pd.get_dummies(df, columns=['strategy'])
    return df

#MACHINE LEARNING: SECONDARY PRICING ENGINE
def train_pricing_model(df):
    print("[INFO] Training ML Pricing Engine (Random Forest Regressor)...")
    
    features = ['vintage', 'tvpi', 'dpi', 'nav_usd'] + [col for col in df.columns if 'strategy_' in col]
    X = df[features]
    y = df['actual_pricing_pct']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"Model R^2 Score (Pricing Accuracy): {r2_score(y_test, predictions):.2f}")
    
    df['predicted_pricing_pct'] = model.predict(X)
    return df

#AGENTIC WORKFLOW: DEAL SCREENING MEMO
def agentic_deal_screener(row):
    """
    Simulates an LLM agent analyzing the ML prediction to generate a recommendation.
    """
    discount = (row['predicted_pricing_pct'] - 1.0) * 100
    
    if row['dpi'] > 0.8 and discount > -10:
        return f"STRONG FIT. The fund's high DPI ({row['dpi']:.2f}) indicates strong cash distribution, offsetting the slight NAV discount. Route to Investment Committee for pricing validation."
    elif discount < -20:
        return f"WARNING. Deep discount predicted ({discount:.1f}%). Flag for intense underlying asset due diligence."
    else:
        return "STANDARD REVIEW. Pricing aligns with vintage and strategy expectations. Proceed with standard LP data room ingestion."

def run_screening_workflow(df):
    print("[INFO] Running Agentic Deal Screener on target portfolio...")
    print("-" * 50)
    
    target = df.iloc[214]
    agent_memo = agentic_deal_screener(target)
    
    strategy_name = [col.replace('strategy_', '') for col in target.index if 'strategy_' in col and target[col] == 1][0]
    
    print("[ARDIAN DEAL SCREENING MEMO]")
    print(f"Target Fund: {target['fund_id']} ({strategy_name}, Vintage {int(target['vintage'])})")
    print(f"Reported NAV: ${target['nav_usd']:,.0f}")
    print(f"Predicted Secondary Price: {target['predicted_pricing_pct']*100:.1f}% of NAV (Discount: {(target['predicted_pricing_pct'] - 1)*100:.1f}%)")
    print(f"Agent Recommendation: {agent_memo}")
    print("-" * 50)

if __name__ == "__main__":
    fund_df = generate_fund_data()
    scored_df = train_pricing_model(fund_df)
    run_screening_workflow(scored_df)
