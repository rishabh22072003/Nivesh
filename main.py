# main.py
import json
import os
import yfinance as yf
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Import your existing stock agent
from stock_agent import MasterAgent

# Import competitor helper
try:
    from competitor_helper import get_competitors, suggest_comparison_metrics, format_competitor_suggestion
    COMPETITOR_HELPER_AVAILABLE = True
except ImportError:
    COMPETITOR_HELPER_AVAILABLE = False
    print("⚠️  Competitor helper not available. Some features may be limited.")


@dataclass
class StockData:
    ticker: str
    analysis_result: Dict[str, Any]
    company_info: Dict[str, Any]
    error: Optional[str] = None


class StockDataCollector:
    
    @staticmethod
    def get_stock_analysis(ticker: str) -> Dict[str, Any]:
        try:
            master_agent = MasterAgent(ticker)
            return master_agent.get_final_verdict()
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Analysis failed: {str(e)}",
                "verdict": {"short_term": "Hold", "long_term": "Hold"},
                "rationale": {"technical": [], "fundamental": []}
            }
    
    @staticmethod
    def get_company_info(ticker: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information for better LLM understanding
            key_info = {
                "basicInfo": {
                    "companyName": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "country": info.get("country", "N/A"),
                    "currency": info.get("currency", "USD"),
                    "exchange": info.get("exchange", "N/A"),
                    "marketCap": info.get("marketCap"),
                    "employees": info.get("fullTimeEmployees")
                },
                "tradingInfo": {
                    "currentPrice": info.get("currentPrice"),
                    "previousClose": info.get("previousClose"),
                    "dayLow": info.get("dayLow"),
                    "dayHigh": info.get("dayHigh"),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "volume": info.get("volume"),
                    "averageVolume": info.get("averageVolume")
                },
                "financialMetrics": {
                    "trailingPE": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "priceToBook": info.get("priceToBook"),
                    "enterpriseValue": info.get("enterpriseValue"),
                    "profitMargins": info.get("profitMargins"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "currentRatio": info.get("currentRatio"),
                    "debtToEquity": info.get("debtToEquity"),
                    "totalRevenue": info.get("totalRevenue"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsQuarterlyGrowth"),
                    "freeCashflow": info.get("freeCashflow")
                },
                "dividendInfo": {
                    "dividendRate": info.get("dividendRate"),
                    "dividendYield": info.get("dividendYield"),
                    "payoutRatio": info.get("payoutRatio"),
                    "exDividendDate": info.get("exDividendDate")
                },
                "analystInfo": {
                    "targetHighPrice": info.get("targetHighPrice"),
                    "targetLowPrice": info.get("targetLowPrice"),
                    "targetMeanPrice": info.get("targetMeanPrice"),
                    "recommendationMean": info.get("recommendationMean"),
                    "recommendationKey": info.get("recommendationKey"),
                    "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions")
                },
                "businessSummary": info.get("longBusinessSummary", "")[:500] + "..." if info.get("longBusinessSummary") else "N/A"
            }
            
            return key_info
            
        except Exception as e:
            return {"error": f"Failed to fetch company info: {str(e)}"}
    
    @classmethod
    def collect_all_data(cls, ticker: str) -> StockData:
        analysis_result = cls.get_stock_analysis(ticker)
        company_info = cls.get_company_info(ticker)
        
        return StockData(
            ticker=ticker,
            analysis_result=analysis_result,
            company_info=company_info
        )


class PromptTemplates:
    
    COMPREHENSIVE_ANALYSIS = ChatPromptTemplate.from_messages([
        ("system", """You are a senior financial advisor with expertise in both technical and fundamental analysis. 

Your role is to provide comprehensive investment recommendations by analyzing:
1. Technical analysis results (RSI, MACD, moving averages, etc.)
2. Fundamental analysis results (PE ratios, profit margins, ROE, etc.) 
3. Company information (sector, industry, market cap, analyst opinions, etc.)

Guidelines for your analysis:
- Provide clear, actionable investment advice
- Explain technical and fundamental reasoning
- Consider company context (sector, size, business model)
- Include both short-term and long-term perspectives
- Mention key risks and opportunities
- Reference analyst opinions when available
- Use professional yet accessible language
- Be objective and balanced in assessment

Structure your response with clear sections and avoid overly technical jargon."""),
        
        ("human", """Please provide a comprehensive investment analysis for {ticker}:

TECHNICAL & FUNDAMENTAL ANALYSIS:
{analysis_json}

COMPANY INFORMATION:
{company_info_json}

Please provide:
1. Executive Summary & Overall Recommendation
2. Technical Analysis Insights
3. Fundamental Analysis Insights  
4. Company & Market Context
5. Risk Assessment
6. Investment Timeline Recommendations
7. Key Factors to Monitor""")
    ])
    
    SPECIFIC_QUESTION = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable financial advisor with access to stock market data. 

When answering questions:
- Be direct, informative, and base answers on the available data
- If asked about competitors but competitor data is not provided, PROACTIVELY:
  1. Identify the company's sector/industry from the provided data
  2. List 3-5 major competitors in that sector
  3. Acknowledge that you don't have their current data but can provide:
     - Known market leaders in the sector
     - Key competitive advantages/disadvantages
     - What metrics would be important to compare
  4. Suggest which competitors the user should analyze
  
- For comparison questions, always provide actionable insights even with limited data
- Use your knowledge of the industry to provide context"""),
        
        ("human", """Stock Data for {ticker}:

ANALYSIS RESULTS:
{analysis_json}

COMPANY INFORMATION:
{company_info_json}

USER QUESTION: {question}

Instructions for answering:
1. First, understand what the user is asking
2. If it's a comparison question and competitor data is NOT provided:
   - Identify the sector/industry from the company info above
   - List known competitors in that sector
   - Explain what key metrics should be compared
   - Provide general competitive positioning based on the available data
   - Suggest the user analyze specific competitors
3. If it's any other question, answer directly using the available data
4. Always be helpful and provide actionable insights

Please provide a comprehensive, helpful answer.""")
    ])
    
    QUICK_SUMMARY = ChatPromptTemplate.from_messages([
        ("system", """Provide a concise investment summary in 2-3 paragraphs. Focus on the most important insights and clear recommendation."""),
        
        ("human", """Summarize the investment case for {ticker}:

Analysis: {analysis_json}
Company Info: {company_info_json}""")
    ])


class FinancialBot:
    """
    GEMINI MODEL OPTIONS (Latest as of Feb 2026):
    
    🏆 RECOMMENDED MODELS:
    
    1. "gemini-2.5-flash" (BEST CHOICE) ⭐
       - Best price-performance ratio
       - Fast processing, low latency
       - Excellent for thinking and agentic tasks
       - 1M token context, 65K output
       - Stable release
    
    2. "gemini-2.5-pro" (MOST ADVANCED)
       - State-of-the-art thinking model
       - Best for complex reasoning
       - Excellent for analyzing large datasets
       - 1M token context, 65K output
       - Stable release
    
    3. "gemini-3-flash-preview" (CUTTING EDGE)
       - Latest generation model
       - Balanced speed and intelligence
       - Frontier capabilities
       - Preview version
    
    4. "gemini-3-pro-preview" (MOST INTELLIGENT)
       - Best multimodal understanding
       - Most powerful for complex analysis
       - Preview version
    
    5. "gemini-2.5-flash-lite" (FASTEST)
       - Optimized for cost and speed
       - High throughput
       - Good for simple tasks
    
    ⚠️ DEPRECATED (Don't use):
    - gemini-2.0-flash (shuts down March 31, 2026)
    - gemini-1.5-flash (NOT SUPPORTED)
    - gemini-1.5-pro (outdated)
    """
    
    def __init__(self, gemini_api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize Financial Bot with Gemini AI
        
        Args:
            gemini_api_key: Your Google Gemini API key
            model: Model name (default: "gemini-2.5-flash")
                  
                  Recommended options:
                  - "gemini-2.5-flash" ⭐ (best overall, stable)
                  - "gemini-2.5-pro" (most advanced, stable)
                  - "gemini-3-flash-preview" (latest, preview)
                  - "gemini-2.5-flash-lite" (fastest, cheapest)
        """
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_api_key,
            temperature=0.3,
            max_tokens=2000
        )
        
        self.data_collector = StockDataCollector()
        self.prompts = PromptTemplates()
        
        # Create chains for different use cases
        self._setup_chains()
        
        print(f"✅ Financial Bot initialized with model: {model}")
    
    def _setup_chains(self):
        """Setup langchain chains for different operations"""
        self.comprehensive_chain = (
            RunnablePassthrough() 
            | self.prompts.COMPREHENSIVE_ANALYSIS 
            | self.llm 
            | StrOutputParser()
        )
        
        self.qa_chain = (
            self.prompts.SPECIFIC_QUESTION 
            | self.llm 
            | StrOutputParser()
        )
        
        self.summary_chain = (
            self.prompts.QUICK_SUMMARY 
            | self.llm 
            | StrOutputParser()
        )
    
    def get_comprehensive_analysis(self, ticker: str) -> str:
        """
        Get comprehensive investment analysis
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Detailed investment analysis
        """
        try:
            # Collect all data
            stock_data = self.data_collector.collect_all_data(ticker)
            
            # Prepare data for LLM
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            # Generate comprehensive analysis
            response = self.comprehensive_chain.invoke({
                "ticker": ticker.upper(),
                "analysis_json": analysis_json,
                "company_info_json": company_info_json
            })
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error analyzing {ticker}: {str(e)}"
    
    def answer_question(self, ticker: str, question: str) -> str:
        """
        Answer specific questions about a stock
        
        Args:
            ticker: Stock ticker symbol
            question: User's question
            
        Returns:
            Answer to the user's question
        """
        try:
            stock_data = self.data_collector.collect_all_data(ticker)
            
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            # Check if question is about competitors
            question_lower = question.lower()
            is_competitor_question = any(word in question_lower for word in 
                ['competitor', 'compare', 'versus', 'vs', 'peer', 'competition', 'rival'])
            
            # If it's a competitor question, enhance the context
            if is_competitor_question and COMPETITOR_HELPER_AVAILABLE:
                # Get sector info
                sector = stock_data.company_info.get('basicInfo', {}).get('sector', 'Unknown')
                industry = stock_data.company_info.get('basicInfo', {}).get('industry', 'Unknown')
                
                # Get known competitors
                competitors = get_competitors(ticker)
                comparison_metrics = suggest_comparison_metrics(sector)
                competitor_info = format_competitor_suggestion(ticker, sector, industry)
                
                enhanced_question = f"""{question}

Additional Context:
{competitor_info}

Based on this information, please:
1. Identify the main competitors mentioned above
2. Explain what makes each competitor important in this sector
3. Suggest key metrics to compare (from the list above)
4. Provide insights on {ticker}'s competitive positioning
5. Recommend which specific competitor stocks to analyze next
"""
                response = self.qa_chain.invoke({
                    "ticker": ticker.upper(),
                    "analysis_json": analysis_json,
                    "company_info_json": company_info_json,
                    "question": enhanced_question
                })
            elif is_competitor_question:
                # Fallback if competitor helper not available
                sector = stock_data.company_info.get('basicInfo', {}).get('sector', 'Unknown')
                industry = stock_data.company_info.get('basicInfo', {}).get('industry', 'Unknown')
                
                enhanced_question = f"""{question}

Additional Context:
- The company operates in the {sector} sector, specifically in {industry}
- Please identify major competitors in this space and provide comparison insights
- Suggest which specific competitor tickers the user should analyze next
- Explain what key metrics would be important to compare in this sector
"""
                response = self.qa_chain.invoke({
                    "ticker": ticker.upper(),
                    "analysis_json": analysis_json,
                    "company_info_json": company_info_json,
                    "question": enhanced_question
                })
            else:
                response = self.qa_chain.invoke({
                    "ticker": ticker.upper(),
                    "analysis_json": analysis_json,
                    "company_info_json": company_info_json,
                    "question": question
                })
            
            return response
            
        except Exception as e:
            return f"Sorry, I couldn't answer your question about {ticker}: {str(e)}"
    
    def compare_stocks_detailed(self, tickers: list) -> str:
        """
        Compare multiple stocks with detailed analysis
        
        Args:
            tickers: List of ticker symbols (2-5 stocks)
            
        Returns:
            Detailed comparison analysis
        """
        try:
            if len(tickers) < 2:
                return "Please provide at least 2 stocks to compare."
            if len(tickers) > 5:
                return "Please limit comparison to 5 stocks maximum for better analysis."
            
            # Collect data for all stocks
            all_data = []
            for ticker in tickers:
                stock_data = self.data_collector.collect_all_data(ticker)
                all_data.append({
                    'ticker': ticker.upper(),
                    'data': stock_data
                })
            
            # Create comparison prompt
            comparison_data = {
                f"stock_{i+1}": {
                    "ticker": data['ticker'],
                    "analysis": data['data'].analysis_result,
                    "info": data['data'].company_info
                }
                for i, data in enumerate(all_data)
            }
            
            comparison_prompt = f"""Compare these stocks and provide investment insights:

{json.dumps(comparison_data, indent=2)}

Please provide:
1. Key Similarities and Differences
2. Valuation Comparison (P/E, P/B, etc.)
3. Financial Health Comparison (margins, ROE, debt, etc.)
4. Technical Outlook Comparison
5. Competitive Advantages of Each
6. Which stock(s) to consider for different investment goals:
   - Short-term trading
   - Long-term growth
   - Dividend income
   - Value investing
7. Risk Assessment for each
8. Final Recommendation with rationale"""

            # Use a generic LLM call for comparison
            response = self.llm.invoke(comparison_prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Sorry, I couldn't complete the comparison: {str(e)}"
    
    def get_quick_summary(self, ticker: str) -> str:
        """
        Get a quick investment summary
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Brief investment summary
        """
        try:
            stock_data = self.data_collector.collect_all_data(ticker)
            
            analysis_json = json.dumps(stock_data.analysis_result, indent=2)
            company_info_json = json.dumps(stock_data.company_info, indent=2)
            
            response = self.summary_chain.invoke({
                "ticker": ticker.upper(),
                "analysis_json": analysis_json,
                "company_info_json": company_info_json
            })
            
            return response
            
        except Exception as e:
            return f"Sorry, I couldn't generate a summary for {ticker}: {str(e)}"
    
    def get_raw_data(self, ticker: str) -> StockData:
        """Get raw stock data for custom processing"""
        return self.data_collector.collect_all_data(ticker)


class FinancialBotInterface:
    """User-friendly interface for the financial bot"""
    
    def __init__(self, gemini_api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize bot interface
        
        Args:
            gemini_api_key: Your Gemini API key
            model: Model to use (default: "gemini-2.5-flash")
                  Options: "gemini-2.5-flash", "gemini-2.5-pro", 
                          "gemini-3-flash-preview", "gemini-2.5-flash-lite"
        """
        self.bot = FinancialBot(gemini_api_key, model=model)
    
    def analyze(self, ticker: str, analysis_type: str = "comprehensive") -> str:
        """
        Main analysis method
        
        Args:
            ticker: Stock ticker
            analysis_type: 'comprehensive', 'quick', or 'summary'
        """
        ticker = ticker.upper()
        
        if analysis_type == "comprehensive":
            return self.bot.get_comprehensive_analysis(ticker)
        elif analysis_type == "quick" or analysis_type == "summary":
            return self.bot.get_quick_summary(ticker)
        else:
            return self.bot.get_comprehensive_analysis(ticker)
    
    def ask(self, ticker: str, question: str) -> str:
        """Ask a specific question about a stock"""
        return self.bot.answer_question(ticker.upper(), question)
    
    def compare_stocks(self, tickers: list) -> str:
        """
        Compare multiple stocks with detailed analysis
        
        Args:
            tickers: List of ticker symbols (2-5 stocks)
            
        Returns:
            Detailed comparison
        """
        return self.bot.compare_stocks_detailed(tickers)
    
    def get_competitors_and_compare(self, ticker: str, competitor_tickers: list = None) -> str:
        """
        Get competitors for a stock and compare them
        
        Args:
            ticker: Main stock ticker
            competitor_tickers: Optional list of competitor tickers
                              If None, will suggest competitors based on sector
            
        Returns:
            Comparison analysis
        """
        if competitor_tickers:
            all_tickers = [ticker] + competitor_tickers
            return self.bot.compare_stocks_detailed(all_tickers)
        else:
            # Get company info to identify sector
            stock_data = self.bot.data_collector.collect_all_data(ticker)
            sector = stock_data.company_info.get('basicInfo', {}).get('sector', 'Unknown')
            industry = stock_data.company_info.get('basicInfo', {}).get('industry', 'Unknown')
            
            suggestion_prompt = f"""For {ticker} which operates in {sector} sector ({industry}), 
please suggest 3-4 major publicly traded competitors that investors should compare it against.
Provide their ticker symbols if known."""
            
            return self.bot.llm.invoke(suggestion_prompt).content


# Example usage and testing
def main():
    """Example usage of the enhanced financial bot"""
    
    # Setup (you need to set your API key)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("Please set your GEMINI_API_KEY environment variable")
        return
    
    # Initialize bot with the latest recommended model
    print("Initializing Financial Bot with gemini-2.5-flash...")
    bot_interface = FinancialBotInterface(GEMINI_API_KEY, model="gemini-2.5-flash")
    
    # Example 1: Comprehensive analysis
    print("\n" + "="*70)
    print("=== COMPREHENSIVE ANALYSIS ===")
    print("="*70)
    ticker = "ITC.NS"
    analysis = bot_interface.analyze(ticker, "comprehensive")
    print(f"\nAnalysis for {ticker}:")
    print(analysis)
    print("\n" + "="*70 + "\n")
    
    # Example 2: Quick summary
    print("=== QUICK SUMMARY ===")
    summary = bot_interface.analyze(ticker, "quick")
    print(f"\nQuick summary for {ticker}:")
    print(summary)
    print("\n" + "="*70 + "\n")
    
    # Example 3: Q&A
    print("=== Q&A EXAMPLES ===")
    questions = [
        "What's the current valuation like?",
        "Should I hold this for retirement?",
        "What are the main risks?",
        "How does this compare to sector peers?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        answer = bot_interface.ask(ticker, question)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()


# Utility functions for integration
class BotUtils:
    """Utility functions for bot integration"""
    
    @staticmethod
    def format_for_web(response: str) -> Dict[str, Any]:
        """Format response for web API"""
        return {
            "status": "success",
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Basic ticker validation"""
        return isinstance(ticker, str) and len(ticker.strip()) > 0
    
    @staticmethod
    def get_supported_analysis_types() -> list:
        """Get list of supported analysis types"""
        return ["comprehensive", "quick", "summary"]
    
    @staticmethod
    def get_available_gemini_models() -> dict:
        """Get list of available Gemini models with descriptions (Feb 2026)"""
        return {
            "gemini-2.5-flash": "⭐ RECOMMENDED - Best price-performance, fast, thinking capable (STABLE)",
            "gemini-2.5-pro": "Most advanced thinking model, best for complex analysis (STABLE)",
            "gemini-3-flash-preview": "Latest generation, balanced speed & intelligence (PREVIEW)",
            "gemini-3-pro-preview": "Most intelligent, best multimodal understanding (PREVIEW)",
            "gemini-2.5-flash-lite": "Fastest, most cost-efficient for simple tasks (STABLE)"
        }


# FastAPI integration example
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Bot API")
bot_interface = FinancialBotInterface(
    os.getenv("GEMINI_API_KEY"), 
    model="gemini-2.5-flash"  # Latest recommended model
)

class AnalysisRequest(BaseModel):
    ticker: str
    analysis_type: str = "comprehensive"

class QuestionRequest(BaseModel):
    ticker: str
    question: str

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        if not BotUtils.validate_ticker(request.ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker")
        
        response = bot_interface.analyze(request.ticker, request.analysis_type)
        return BotUtils.format_for_web(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not BotUtils.validate_ticker(request.ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker")
        
        response = bot_interface.ask(request.ticker, request.question)
        return BotUtils.format_for_web(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""