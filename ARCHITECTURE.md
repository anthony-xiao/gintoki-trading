graph TD
    A[Polygon.io Real-time Data] -->|Done| B(Node.js Data Ingestor)
    B -->|Done| C{AWS S3 Historical Storage}
    C -->|Partial| D[Python Training Cluster]
    D -->|Not Started| E[SHAP Feature Optimizer]
    E -->|Not Started| F[LSTM Model Factory]
    F -->|Not Started| G[Alpaca Execution Engine]
    G -->|Not Started| H[Performance Monitor]
    H -->|Not Started| D

Here's a straightforward explanation of what the training system does and the data it uses:

---

### **What the Model is Doing**
**In Simple Terms:**  
The model learns to predict whether the price of a stock will go **UP** or **DOWN** in the next few minutes, based on patterns it sees in historical trading data. It's like teaching a computer to recognize the hidden signals in market behavior that often lead to price movements.

---

### **Key Components Explained**

1. **Input Data**  
   - **RSI (30-70 Scale):** How "overbought" or "oversold" a stock is  
     *(Like a fuel gauge for buying/selling pressure)*
   - **OBV:** Whether big trades are happening on price increases vs decreases  
     *(Smart money tracking)*
   - **VWAP:** The average price traders are paying right now  
     *(Institutional benchmark)*
   - **Volume:** How many shares are being traded  

2. **Prediction Target**  
   - **Goal:** Predict if the next 1-minute closing price will be HIGHER than the current price  
   - **Output:** Probability between 0-1 (0.65 = 65% chance price will rise)

3. **Learning Process**  
   - Looks at the last **60 minutes** of trading data (60 data points)  
   - Finds hidden patterns in how RSI/OBV/VWAP/Volume interact  
   - Adjusts its internal "rules" through 20 rounds of pattern review  

4. **Safety Checks**  
   - Tests itself on unseen data to avoid "cheating"  
   - Uses SHAP analysis to only keep important features  
   - Saves best-performing version to AWS S3

---

### **Data Specifications**

| Aspect              | Detail                                                                 |
|---------------------|-----------------------------------------------------------------------|
| **Source**           | Processed tick data in S3 (`processed/ticks/` directory)             |
| **Time Frequency**  | 1-minute intervals (60 data points = 1 hour of market activity)       |
| **Data Volume**      | ~3.5M minutes/year per stock (250 trading days * 6.5hrs * 60min)     |
| **Update Schedule** | Daily training at market close (4pm EST) using that day's data       |

---

### **Real-World Analogy**  
Imagine training a weather forecaster to predict rain:  
1. **Inputs:** Humidity, wind speed, pressure trends (last 60 mins)  
2. **Output:** Chance of rain in next 5 mins  
3. **Improvement:** Learn which factors matter most (humidity > wind)  

The trading model works similarly, but for price movements instead of weather.

---

### **Why This Approach Works**  
Based on our initial research of successful algorithmic strategies:
1. **LSTMs** excel at finding patterns in time-series data (price movements)  
2. **1-minute bars** capture intraday swings without market noise  
3. **SHAP filtering** prevents overfitting to irrelevant signals  
4. **Walk-forward validation** mimics real trading conditions  

---

**Need any part explained differently or want to adjust the strategy?**  
For example:  
- Add daily/weekly trends for longer-term context  
- Include order book depth for better OBV calculation  
- Switch to 5-minute bars for slower-moving stocks

Sources
