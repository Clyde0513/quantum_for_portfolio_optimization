
"""
V Data Loader for Quantum Portfolio Optimization

This module loads and processes V bond portfolio data from Excel files,
converting it into the format required by the quantum portfolio optimization algorithm.

The data comes from V's fund containing:
- 2,629 individual bond positions (primarily corporate bonds)
- Real market values and position sizes  
- Comprehensive risk measures: Duration vectors (KRD), credit spreads (OAS), credit ratings
- Sector classifications and issuer information
- Portfolio weights derived from actual market value positions

Key Financial Terms in Dataset:
- OAD (Option-Adjusted Duration): Interest rate sensitivity measure (avg: 6.0 years)
- OAS (Option-Adjusted Spread): Credit spread over treasuries (avg: 79 bps) 
- KRD (Key Rate Duration): Interest rate risk at specific maturity points (3m, 1y, 2y, 3y, 5y, 7y, 10y, 15y, 20y, 25y, 30y)
- Credit Quality: Investment grade bonds (A: 33%, BBB: 52%, AAA/AA: 15%)
- Sectors: Industrial (48%), Financial (28%), Utilities (12%), Government (11%)
- Market Value: Total portfolio ~$50.7B, individual positions range from -$44M to +$357M
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

def load_v_portfolio_data(file_path: str = "data_assets_dictionary.xlsx", n_assets: int = 50) -> Dict:
    """
    Load V bond portfolio data from Excel file.
    
    The dataset contains 2,629 bond positions from V fund with:
    - Real market values for portfolio weight calculation
    - Duration risk vectors (KRD) for interest rate sensitivity modeling
    - Credit spreads (OAS) and ratings for credit risk assessment
    - Sector classifications for diversification constraints
    
    Args:
        file_path: Path to the V data Excel file (4.9MB with 278 columns)
        n_assets: Number of assets to select for optimization (default: 50 for quantum feasibility)
        
    Returns:
        Dictionary containing portfolio data with keys:
        - 'returns': Expected returns derived from credit spreads and duration
        - 'risks': Risk measures combining duration and credit risk
        - 'correlations': Correlation matrix estimated from sector/credit clustering
        - 'asset_names': Asset identifiers (ISIN codes and tickers)
        - 'weights': Current portfolio weights from market values
        - 'additional_data': Selected subset of full dataset for analysis
        - 'risk_factors': Detailed risk decomposition by duration buckets
        - 'sector_info': Sector and credit quality classifications
        - 'portfolio_info': Summary statistics of the portfolio
    """
    
    try:
        # Load the Excel file using openpyxl engine
        print("Loading V bond portfolio data...")
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Successfully loaded {len(df)} bond positions from V portfolio")
        print(f"Dataset dimensions: {df.shape[0]} assets x {df.shape[1]} features")

        # Filter for corporate bonds only (main focus of V fund)
        bond_data = df[df['secGroup'] == 'BND'].copy()
        print(f"Filtered to {len(bond_data)} bond positions")
        
        # Select top holdings by market value for quantum optimization
        # This ensures we work with the most significant positions
        bond_data = bond_data.nlargest(n_assets, 'fund_enriched.mktValue')
        print(f"Selected top {len(bond_data)} holdings by market value for optimization")
        
        # Extract portfolio metadata
        portfolio_data = extract_portfolio_info(bond_data)
        
        # Process the financial data for quantum optimization
        returns = extract_returns(bond_data)
        risks = extract_risks(bond_data) 
        correlations = estimate_correlations(bond_data)
        weights = calculate_current_weights(bond_data)
        asset_names = get_asset_identifiers(bond_data)
        risk_factors = extract_risk_factors(bond_data)
        sector_info = extract_sector_classifications(bond_data)
        
        print(f"\nPortfolio Analysis Summary:")
        print(f"  Fund: ${portfolio_data['total_market_value']:,.0f}")
        print(f"  Average Duration: {portfolio_data['avg_duration']:.2f} years")
        print(f"  Average Credit Spread: {portfolio_data['avg_credit_spread']:.0f} basis points")
        print(f"  Expected Returns: [{returns.min():.3f}, {returns.max():.3f}]")
        print(f"  Risk Measures: [{risks.min():.3f}, {risks.max():.3f}]")
        print(f"  Sector Distribution: {list(portfolio_data['sector_distribution'].keys())[:3]}")
        
        return {
            'returns': returns,
            'risks': risks,
            'correlations': correlations,
            'asset_names': asset_names,
            'weights': weights,
            'additional_data': bond_data,
            'portfolio_info': portfolio_data,
            'risk_factors': risk_factors,
            'sector_info': sector_info,
            'data_source': 'v_real'
        }
        
    except Exception as e:
        print(f"Error loading V data: {e}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_fallback(n_assets)

def extract_portfolio_info(df: pd.DataFrame) -> Dict:
    """Extract portfolio-level information and statistics."""
    
    total_mv = df['fund_enriched.mktValue'].sum()
    
    return {
        'fund_name': df['portfolioName'].iloc[0] if 'portfolioName' in df.columns else 'V Fund',
        'n_assets': len(df),
        'currency': df['ccy'].iloc[0] if 'ccy' in df.columns else 'USD',
        'total_market_value': total_mv,
        'avg_duration': df['oad'].mean(),
        'avg_credit_spread': df['oas'].mean(),
        'avg_coupon': df['cpn'].mean(),
        'credit_quality_distribution': df['enriched.creditQualityBuckets'].value_counts().to_dict(),
        'sector_distribution': df['sodraw.filterLevel2'].value_counts().to_dict(),
        'largest_holding_pct': df['fund_enriched.mktValue'].max() / total_mv * 100,
        'avg_maturity_years': df['Maturity Years'].mean() if 'Maturity Years' in df.columns else None
    }

def extract_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Extract expected returns from bond characteristics.
    
    For bonds, expected return is primarily driven by:
    1. Credit spread (OAS) - compensation for credit risk
    2. Duration - interest rate sensitivity
    3. Current yield environment
    
    We use the Option-Adjusted Spread (OAS) as the primary return driver
    since it represents the excess return over risk-free rate.
    """
    
    # Base risk-free rate (approximate current 10-year treasury)
    risk_free_rate = 0.045  # 4.5%
    
    # Extract credit spreads (Option-Adjusted Spread) in basis points
    credit_spreads = df['oas'].fillna(df['oas'].mean()).values
    
    # Convert basis points to decimal (divide by 10,000)
    credit_spread_decimal = credit_spreads / 10000
    
    # Expected return = Risk-free rate + Credit spread
    # Add small adjustment for duration risk
    durations = df['oad'].fillna(df['oad'].mean()).values
    duration_adjustment = durations * 0.001  # Small duration premium
    
    expected_returns = risk_free_rate + credit_spread_decimal + duration_adjustment
    
    # Ensure reasonable bounds for bond returns (1% to 8%)
    expected_returns = np.clip(expected_returns, 0.01, 0.08)
    
    print(f"  Returns calculated from OAS (credit spreads): {credit_spreads.mean():.0f} bps average")
    
    return expected_returns

def extract_risks(df: pd.DataFrame) -> np.ndarray:
    """
    Extract risk measures from bond duration and spread characteristics.
    
    Bond risk comes from two main sources:
    1. Interest rate risk (measured by duration)
    2. Credit risk (measured by spread volatility)
    
    We combine Option-Adjusted Duration (OAD) with credit spread volatility
    to create a comprehensive risk measure.
    """
    
    # Primary risk measure: Option-Adjusted Duration (interest rate sensitivity)
    durations = df['oad'].fillna(df['oad'].mean()).values
    
    # Secondary risk: Credit spread volatility (estimated from OAS)
    credit_spreads = df['oas'].fillna(df['oas'].mean()).values
    
    # Convert duration to annual volatility estimate
    # Typical bond duration volatility: Duration × 0.01 (1% rate change)
    duration_risk = durations * 0.01
    
    # Credit risk estimate: Higher spreads = higher credit risk
    # Scale credit spreads to volatility range (basis points to decimal)
    credit_risk = (credit_spreads / 10000) * 0.5  # Scale factor for credit volatility
    
    # Combined risk measure (duration risk dominates for IG bonds)
    total_risk = np.sqrt(duration_risk**2 + credit_risk**2)
    
    # Ensure reasonable risk bounds (0.5% to 15% annual volatility)
    total_risk = np.clip(total_risk, 0.005, 0.15)
    
    print(f"  Risk calculated from duration (avg: {durations.mean():.2f} years) and credit spreads")
    
    return total_risk

def estimate_correlations(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate correlation matrix based on sector and credit quality clustering.
    
    Bond correlations are driven by:
    1. Sector exposure (industrial, financial, utilities)
    2. Credit quality (AAA, AA, A, BBB)
    3. Duration bucket (short, medium, long-term)
    4. General market factor
    """
    
    n_assets = len(df)
    corr_matrix = np.eye(n_assets)
    
    # Extract sector and credit quality information
    sectors = df['sodraw.filterLevel2'].fillna('Unknown').values
    credit_qualities = df['enriched.creditQualityBuckets'].fillna('NR').values
    durations = df['oad'].fillna(df['oad'].mean()).values
    
    # Define correlation structure
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation = 0.05  # Base market correlation
            
            # Same sector correlation boost
            if sectors[i] == sectors[j] and sectors[i] != 'Unknown':
                correlation += 0.25
            
            # Same credit quality correlation boost  
            if credit_qualities[i] == credit_qualities[j] and credit_qualities[i] != 'NR':
                correlation += 0.15
            
            # Similar duration correlation boost
            duration_diff = abs(durations[i] - durations[j])
            if duration_diff < 2.0:  # Within 2 years duration
                correlation += 0.10
            
            # Cap maximum correlation
            correlation = min(correlation, 0.85)
            
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
    
    # Ensure positive semi-definite matrix
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Normalize diagonal to 1
    np.fill_diagonal(corr_matrix, 1.0)
    
    avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    print(f"  Correlation matrix estimated from sector/credit clustering (avg: {avg_correlation:.3f})")
    
    return corr_matrix

def calculate_current_weights(df: pd.DataFrame) -> np.ndarray:
    """Calculate portfolio weights from current market values."""
    
    market_values = df['fund_enriched.mktValue'].values
    total_value = market_values.sum()
    
    weights = market_values / total_value
    
    print(f"  Weights calculated from market values (largest: {weights.max():.3f})")
    
    return weights

def get_asset_identifiers(df: pd.DataFrame) -> List[str]:
    """Extract asset identifiers, preferring ISIN codes then tickers."""
    
    if 'isin' in df.columns:
        asset_names = df['isin'].fillna('UNKNOWN').values
    elif 'issuerTicker' in df.columns:
        asset_names = df['issuerTicker'].fillna('UNKNOWN').values
    else:
        asset_names = [f'Asset_{i:03d}' for i in range(len(df))]
    
    return list(asset_names)

def extract_risk_factors(df: pd.DataFrame) -> Dict:
    """Extract detailed risk factor decomposition."""
    
    # Key Rate Duration (KRD) risk factors
    krd_columns = [col for col in df.columns if col.startswith('krd') and 'Ctr' not in col]
    
    risk_factors = {}
    for col in krd_columns:
        if col in df.columns:
            risk_factors[col] = df[col].fillna(0).values
    
    risk_factors['credit_spreads'] = df['oas'].fillna(df['oas'].mean()).values
    risk_factors['durations'] = df['oad'].fillna(df['oad'].mean()).values
    
    return risk_factors

def extract_sector_classifications(df: pd.DataFrame) -> Dict:
    """Extract sector and classification information."""
    
    return {
        'sectors': df['sodraw.filterLevel2'].fillna('Unknown').values,
        'credit_qualities': df['enriched.creditQualityBuckets'].fillna('NR').values,
        'industries': df['sodraw.filterLevel3'].fillna('Unknown').values,
        'issuers': df['issuerTicker'].fillna('Unknown').values,
        'countries': df['issuerCountry'].fillna('Unknown').values
    }

def generate_synthetic_fallback(n_assets: int = 50) -> Dict:
    """Generate synthetic but realistic bond portfolio data as fallback."""
    
    print(f"Generating synthetic bond portfolio with {n_assets} assets...")
    
    # Realistic bond sectors and their proportions
    sectors = ['Industrial'] * int(n_assets * 0.48) + \
              ['Financial'] * int(n_assets * 0.28) + \
              ['Utilities'] * int(n_assets * 0.12) + \
              ['Government'] * (n_assets - int(n_assets * 0.88))
    
    # Credit quality distribution
    credit_qualities = ['BBB'] * int(n_assets * 0.52) + \
                      ['A'] * int(n_assets * 0.33) + \
                      ['AAA/AA'] * (n_assets - int(n_assets * 0.85))
    
    # Generate realistic bond characteristics
    durations = np.random.uniform(4.0, 8.0, n_assets)  # 4-8 year duration
    credit_spreads = np.random.uniform(50, 120, n_assets)  # 50-120 bps spreads
    
    # Calculate returns and risks
    returns = 0.045 + (credit_spreads / 10000) + (durations * 0.001)
    risks = durations * 0.01 + (credit_spreads / 10000) * 0.5
    
    # Build correlation matrix
    correlations = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = 0.05  # Base correlation
            if sectors[i] == sectors[j]:
                corr += 0.25
            if credit_qualities[i] == credit_qualities[j]:
                corr += 0.15
            
            correlations[i,j] = correlations[j,i] = min(corr, 0.85)
    
    # Equal weights for synthetic portfolio
    weights = np.ones(n_assets) / n_assets
    
    asset_names = [f'SYNTH_BOND_{i:03d}' for i in range(n_assets)]
    
    portfolio_info = {
        'fund_name': 'Synthetic Bond Portfolio',
        'n_assets': n_assets,
        'currency': 'USD',
        'total_market_value': 50_000_000_000,  # $50B synthetic
        'avg_duration': durations.mean(),
        'avg_credit_spread': credit_spreads.mean(),
        'sector_distribution': dict(zip(*np.unique(sectors, return_counts=True))),
        'credit_quality_distribution': dict(zip(*np.unique(credit_qualities, return_counts=True)))
    }
    
    return {
        'returns': returns,
        'risks': risks,
        'correlations': correlations,
        'asset_names': asset_names,
        'weights': weights,
        'additional_data': None,
        'portfolio_info': portfolio_info,
        'risk_factors': {'durations': durations, 'credit_spreads': credit_spreads},
        'sector_info': {'sectors': sectors, 'credit_qualities': credit_qualities},
        'data_source': 'synthetic_realistic'
    }

# Integration function for quantum optimization code
def get_quantum_optimization_data(n_assets: int = 20) -> Tuple:
    """
    Load V data and return in format expected by quantum optimization code.
    
    Returns:
        Tuple of (n_assets, returns, risks, correlations, asset_names, data_source)
    """
    
    data = load_v_portfolio_data(n_assets=n_assets)
    
    return (
        len(data['returns']),
        data['returns'],
        data['risks'], 
        data['correlations'],
        data['asset_names'],
        data['data_source']
    )

if __name__ == "__main__":
    # Test the data loader
    print("Testing V Data Loader...")
    print("=" * 50)

    data = load_v_portfolio_data(n_assets=20)

    print(f"\nData Loading Results:")
    print(f"  Source: {data['data_source']}")
    print(f"  Assets: {len(data['returns'])}")
    print(f"  Returns range: [{data['returns'].min():.3f}, {data['returns'].max():.3f}]")
    print(f"  Risk range: [{data['risks'].min():.3f}, {data['risks'].max():.3f}]")
    print(f"  Sample assets: {data['asset_names'][:5]}")
    print(f"  Portfolio: {data['portfolio_info']['fund_name']}")
    print(f"  Average duration: {data['portfolio_info']['avg_duration']:.2f} years")
    print(f"  Average spread: {data['portfolio_info']['avg_credit_spread']:.0f} bps")
