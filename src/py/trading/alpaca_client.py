import os
import logging
from typing import Dict, Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AlpacaTradingClient:
    def __init__(self, paper: bool = True):
        """Initialize Alpaca trading client"""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
            
        self.client = TradingClient(
            self.api_key,
            self.api_secret,
            paper=paper
        )
        
        # Get account information
        self.account = self.client.get_account()
        logger.info(f"Connected to Alpaca account: {self.account.id}")
        
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol"""
        try:
            positions = self.client.get_all_positions()
            for position in positions:
                if position.symbol == symbol:
                    return {
                        'symbol': position.symbol,
                        'qty': float(position.qty),
                        'avg_entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'market_value': float(position.market_value)
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
            
    def execute_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[Dict]:
        """Execute a market order"""
        try:
            # Prepare order request
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.client.submit_order(order_data)
            
            # Wait for order to complete
            order.wait()
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'status': order.status,
                'filled_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
        except Exception as e:
            logger.error(f"Error executing order for {symbol}: {str(e)}")
            return None
            
    def get_account_summary(self) -> Dict:
        """Get account summary including cash and positions"""
        try:
            account = self.client.get_account()
            positions = self.client.get_all_positions()
            
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value)
                    }
                    for pos in positions
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {} 