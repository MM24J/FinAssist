## Final Implementation

### Working Features
- **RAG System**: Responds to financial advice questions using vector search and knowledge base
- **Budget Analysis**: Analyzes real transaction data from CSV files for spending insights
- **Investment Tracking**: Retrieves live stock data via yfinance API
- **Smart Classification**: Routes questions to appropriate analysis engines

### Deployment
The application is deployed using Gradio with a web interface accessible at 'app.py'.

### Sample Interactions

**Financial Advice (RAG System):**

User: "How can I lower my restaurant bill?"
FinAssist:

• Set a weekly dining cap and track it. (e.g., set a weekly dining cap of $40-60 and track spend).

• Meal prep 2 lunches per week to cut impulse orders.

• Move coffee/fast food into a shared "Dining Out" cap. (e.g., set a weekly dining cap of $40-60 and track spend).

• Groceries up + dining up → overbuying & eating out. Plan meals, shop with a list.

• Groceries up + dining down → home cooking trend; check waste/spoilage.

Source: FinAssist KB

**Budget Analysis (Real Data):**

User: "Show me my top 3 over-budget categories in 2019-09"
FinAssist:
--- Budget Summary for September 2019 ---

Top 3 over-budget categories:

• Credit Card Payment: +$3,860.14 (Actual $3,960.14 vs Budget $100.00)

• Mortgage & Rent: +$1,000.00 (Actual $1,100.00 vs Budget $100.00)

• Utilities: +$25.00 (Actual $125.00 vs Budget $100.00)

Top 3 under-budget categories:

• Shopping: $-164.62 (Actual $85.38 vs Budget $250.00)

• Groceries: $-160.76 (Actual $139.24 vs Budget $300.00)

• Music: $-89.31 (Actual $10.69 vs Budget $100.00)

**Investment Tracking (Live Data):**
User: "Compare TSLA vs SPY over 6m

--- Investment Summary for 6mo ---

TSLA: +61.24% (Start $273.13 → End $440.40; Last $440.40)

SPY: +17.38% (Start $563.85 → End $661.82; Last $661.82)
