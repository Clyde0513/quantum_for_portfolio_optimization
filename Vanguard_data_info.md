# Vanguard Data Information

## **Dataset Analysis Summary**

### **Portfolio Characteristics (VCIT Fund)**

- **Total Assets**: 2,629 individual bond positions
- **Fund Size**: ~$50.7 billion market value
- **Primary Focus**: Corporate intermediate-term bonds
- **Currency**: USD denominated

### **Key Financial Metrics Understood**

1. **OAD (Option-Adjusted Duration)**: 6.0 years average
   - Measures interest rate sensitivity
   - Range: 0 to 19.2 years

2. **OAS (Option-Adjusted Spread)**: 79 basis points average  
   - Credit spread over risk-free rate
   - Primary driver of expected returns
   - Range: -14 to 294 basis points

3. **Credit Quality Distribution**:
   - BBB rated: 52% (1,372 bonds)
   - A rated: 33% (863 bonds)  
   - AAA/AA rated: 15% (389 bonds)

4. **Sector Distribution**:
   - Industrial: 48% (1,273 bonds)
   - Financial: 28% (735 bonds)
   - Utilities: 12% (320 bonds)
   - Government: 11% (286 bonds)

### **Risk Measures (KRD - Key Rate Durations)**

- **krd5y**: 1.38 average (5-year rate sensitivity)
- **krd7y**: 2.45 average (7-year rate sensitivity)  
- **krd10y**: 1.37 average (10-year rate sensitivity)
- Full duration vector available: 3m, 1y, 2y, 3y, 5y, 7y, 10y, 15y, 20y, 25y, 30y

## **Data Loader Features**

### **Real Data Processing**

- Loads 2,629 actual Vanguard bond positions
- Filters to top holdings by market value for quantum feasibility
- Calculates expected returns from credit spreads (OAS)
- Derives risk measures from duration and credit volatility
- Estimates correlations from sector/credit clustering
- Extracts real portfolio weights from market values

### **Financial Accuracy**

- Expected returns = Risk-free rate + Credit spread + Duration premium
- Risk measures = √(Duration risk² + Credit risk²)
- Correlations based on sector (0.25 boost), credit (0.15 boost), duration (0.10 boost)
- All parameters within realistic bond market ranges

## **Integration Instructions**

### **Integration** (will explore this more)

Replace the data loading section in your quantum code with:

```python
from vanguard_data_loader import get_quantum_optimization_data
n, expected_returns, risk_measures, correlation_matrix, asset_names, data_source = get_quantum_optimization_data(n_assets=30)
```

### **Full Integration Example**

```python
# Load real Vanguard data
from vanguard_data_loader import load_vanguard_portfolio_data

# Get comprehensive portfolio data
portfolio_data = load_vanguard_portfolio_data(n_assets=50)

# Extract for quantum optimization
returns = portfolio_data['returns']
risks = portfolio_data['risks'] 
correlations = portfolio_data['correlations']
weights = portfolio_data['weights']
asset_names = portfolio_data['asset_names']

print(f"Portfolio: {portfolio_data['portfolio_info']['fund_name']}")
print(f"Duration: {portfolio_data['portfolio_info']['avg_duration']:.2f} years")
print(f"Credit Spread: {portfolio_data['portfolio_info']['avg_credit_spread']:.0f} bps")
```

## **Verification Results**

- Module imports correctly
- Excel file loads (2,629 × 278 dimensions)
- Data processing functions work
- Real Vanguard data extracted:
  - Returns: 4.9% to 7.4% (realistic bond yields)
  - Risks: 4.4% to 7.0% (appropriate volatilities)
  - 20-50 assets ready for quantum optimization
