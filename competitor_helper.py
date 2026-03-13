"""
Competitor Analysis Helper
Helps identify and compare competitor stocks
"""

# Common competitor mappings for Indian stocks
INDIAN_STOCK_COMPETITORS = {
    # FMCG Sector
    "ITC.NS": ["HINDUNILVR.NS", "DABUR.NS", "BRITANNIA.NS", "NESTLEIND.NS"],
    "HINDUNILVR.NS": ["ITC.NS", "DABUR.NS", "BRITANNIA.NS", "NESTLEIND.NS"],
    "DABUR.NS": ["ITC.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "PATANJALI.NS"],
    "BRITANNIA.NS": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "TATACONSUM.NS"],
    
    # IT Sector
    "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "WIPRO.NS": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
    "HCLTECH.NS": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    
    # Banking Sector
    "HDFCBANK.NS": ["ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
    "ICICIBANK.NS": ["HDFCBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
    "SBIN.NS": ["HDFCBANK.NS", "ICICIBANK.NS", "PNB.NS", "BANKBARODA.NS"],
    
    # Automotive
    "MARUTI.NS": ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "TATAMOTORS.NS": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "EICHERMOT.NS"],
    "M&M.NS": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS"],
    
    # Telecom
    "BHARTIARTL.NS": ["RJIO.NS", "IDEA.NS"],
    "IDEA.NS": ["BHARTIARTL.NS", "RJIO.NS"],
    
    # Oil & Gas
    "RELIANCE.NS": ["ONGC.NS", "IOC.NS", "BPCL.NS"],
    "ONGC.NS": ["RELIANCE.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"],
    
    # Pharma
    "SUNPHARMA.NS": ["DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    "DRREDDY.NS": ["SUNPHARMA.NS", "CIPLA.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    "CIPLA.NS": ["SUNPHARMA.NS", "DRREDDY.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    
    # Cement
    "ULTRACEMCO.NS": ["GRASIM.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS"],
    "GRASIM.NS": ["ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS"],
    
    # Steel
    "TATASTEEL.NS": ["JSWSTEEL.NS", "SAIL.NS", "HINDALCO.NS"],
    "JSWSTEEL.NS": ["TATASTEEL.NS", "SAIL.NS", "JINDALSTEL.NS"],
}

# US Stock competitors
US_STOCK_COMPETITORS = {
    # Tech Giants
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
    "GOOGL": ["AAPL", "MSFT", "META", "AMZN"],
    "META": ["GOOGL", "SNAP", "TWTR", "PINS"],
    "AMZN": ["WMT", "TGT", "EBAY", "SHOP"],
    
    # Auto
    "TSLA": ["F", "GM", "RIVN", "LCID"],
    "F": ["GM", "TSLA", "TM", "STLA"],
    "GM": ["F", "TSLA", "TM", "STLA"],
    
    # Finance
    "JPM": ["BAC", "WFC", "C", "GS"],
    "BAC": ["JPM", "WFC", "C", "USB"],
    "WFC": ["JPM", "BAC", "C", "USB"],
    
    # Retail
    "WMT": ["TGT", "COST", "AMZN", "DG"],
    "TGT": ["WMT", "COST", "KSS", "M"],
    "COST": ["WMT", "TGT", "BJ", "SAM"],
    
    # Pharma
    "PFE": ["JNJ", "MRK", "ABBV", "BMY"],
    "JNJ": ["PFE", "MRK", "ABBV", "UNH"],
    "MRK": ["PFE", "JNJ", "LLY", "GSK"],
    
    # Energy
    "XOM": ["CVX", "COP", "SLB", "BP"],
    "CVX": ["XOM", "COP", "OXY", "PSX"],
    
    # Aerospace
    "BA": ["LMT", "RTX", "GD", "NOC"],
    "LMT": ["BA", "RTX", "GD", "NOC"],
}

# Sector-based generic competitors
SECTOR_COMPETITORS = {
    "FMCG": ["Consumer goods companies", "Food & beverage", "Personal care"],
    "IT": ["Software services", "IT consulting", "Technology"],
    "Banking": ["Commercial banks", "Financial services", "NBFCs"],
    "Automotive": ["Auto manufacturers", "Auto components", "EV companies"],
    "Pharma": ["Pharmaceutical companies", "Healthcare", "Biotech"],
    "Telecom": ["Telecommunication services", "Network providers"],
    "Energy": ["Oil & gas", "Renewables", "Utilities"],
    "Retail": ["E-commerce", "Traditional retail", "Hypermarkets"],
}


def get_competitors(ticker, max_competitors=4):
    """
    Get competitor tickers for a given stock
    
    Args:
        ticker: Stock ticker symbol
        max_competitors: Maximum number of competitors to return
        
    Returns:
        List of competitor ticker symbols
    """
    ticker = ticker.upper()
    
    # Check Indian stocks first
    if ticker in INDIAN_STOCK_COMPETITORS:
        competitors = INDIAN_STOCK_COMPETITORS[ticker][:max_competitors]
        return competitors
    
    # Check US stocks
    if ticker in US_STOCK_COMPETITORS:
        competitors = US_STOCK_COMPETITORS[ticker][:max_competitors]
        return competitors
    
    # If not found, return empty list
    return []


def get_sector_info(sector):
    """
    Get information about competitors in a sector
    
    Args:
        sector: Sector name
        
    Returns:
        Dictionary with sector competitor info
    """
    sector_upper = sector.upper()
    
    for key in SECTOR_COMPETITORS:
        if key.upper() in sector_upper or sector_upper in key.upper():
            return {
                "sector": key,
                "competitor_types": SECTOR_COMPETITORS[key]
            }
    
    return {"sector": sector, "competitor_types": ["Similar companies in the sector"]}


def suggest_comparison_metrics(sector):
    """
    Suggest important metrics to compare based on sector
    
    Args:
        sector: Sector name
        
    Returns:
        List of important comparison metrics
    """
    sector_metrics = {
        "FMCG": [
            "Revenue Growth",
            "Profit Margins",
            "Market Share",
            "Brand Value",
            "Distribution Network",
            "EBITDA Margins"
        ],
        "IT": [
            "Revenue Growth",
            "Employee Count",
            "Client Base",
            "Operating Margins",
            "R&D Investment",
            "Attrition Rate"
        ],
        "Banking": [
            "NPA Ratio",
            "Net Interest Margin (NIM)",
            "CASA Ratio",
            "Return on Assets (ROA)",
            "Capital Adequacy Ratio",
            "Loan Growth"
        ],
        "Automotive": [
            "Sales Volume",
            "Market Share",
            "EBITDA Margins",
            "R&D Investment",
            "Debt to Equity",
            "Production Capacity"
        ],
        "Pharma": [
            "Revenue Growth",
            "R&D as % of Revenue",
            "EBITDA Margins",
            "Product Pipeline",
            "Export Revenue",
            "Generic vs Branded Mix"
        ],
    }
    
    for key in sector_metrics:
        if key.upper() in sector.upper() or sector.upper() in key.upper():
            return sector_metrics[key]
    
    # Default metrics
    return [
        "Revenue Growth",
        "Profit Margins",
        "Return on Equity",
        "Debt to Equity",
        "P/E Ratio",
        "Market Cap"
    ]


def format_competitor_suggestion(ticker, sector, industry):
    """
    Format a nice suggestion message about competitors
    
    Args:
        ticker: Stock ticker
        sector: Sector name
        industry: Industry name
        
    Returns:
        Formatted suggestion string
    """
    competitors = get_competitors(ticker)
    metrics = suggest_comparison_metrics(sector)
    
    if competitors:
        suggestion = f"""
📊 **Competitor Analysis for {ticker}**

**Sector**: {sector}
**Industry**: {industry}

**Main Competitors**:
{chr(10).join([f"• {comp}" for comp in competitors])}

**Key Metrics to Compare**:
{chr(10).join([f"• {metric}" for metric in metrics])}

**Next Steps**:
1. Analyze these competitor stocks individually
2. Compare their financial metrics
3. Evaluate competitive advantages
4. Consider market positioning

Would you like me to compare {ticker} with any specific competitors?
"""
    else:
        suggestion = f"""
📊 **Competitor Analysis for {ticker}**

**Sector**: {sector}
**Industry**: {industry}

I don't have pre-defined competitors for this stock in my database, but here's how to find them:

**Key Metrics to Compare in {sector}**:
{chr(10).join([f"• {metric}" for metric in metrics])}

**How to Find Competitors**:
1. Look for companies in the same sector and industry
2. Check companies with similar market cap
3. Look at analyst reports for peer comparisons
4. Check stock screeners for similar companies

Once you identify competitors, you can enter their tickers above for a detailed comparison!
"""
    
    return suggestion


# Example usage
if __name__ == "__main__":
    # Test with Indian stock
    print("Testing ITC.NS:")
    competitors = get_competitors("ITC.NS")
    print(f"Competitors: {competitors}")
    
    metrics = suggest_comparison_metrics("FMCG")
    print(f"Comparison metrics: {metrics}")
    
    suggestion = format_competitor_suggestion("ITC.NS", "FMCG", "Tobacco & FMCG")
    print(suggestion)
    
    print("\n" + "="*60 + "\n")
    
    # Test with US stock
    print("Testing AAPL:")
    competitors = get_competitors("AAPL")
    print(f"Competitors: {competitors}")
    
    suggestion = format_competitor_suggestion("AAPL", "Technology", "Consumer Electronics")
    print(suggestion)