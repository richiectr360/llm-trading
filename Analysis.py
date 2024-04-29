from yahooquery import Ticker
from datetime import datetime
import yfinance as yf
import requests

class Analysis:
	def __init__(self):
		self.query = None
		
	def company(self, string):
		lis = list(string.split(" "))
		length = len(lis)
		return lis[length-1]
		
	def get_stock_history(self, symbol, date):
		ticker = yf.Ticker(symbol)
		data = ticker.history(start = "2019-01-01", end = date)
		return data

	def get_latest_price(self, data):
		latest_price = data['Close'].iloc[-1]
		return latest_price
		
	def get_gnews_api(self):
		url = "https://gnews.io/api/v4/top-headlines?lang=en&token=9711f50cd8d81cc3f4406fb08839b378"
		response = requests.get(url)
		news = response.json()
		return news

	def get_gnews_api_spec(self, search_term):
		url = f"https://gnews.io/api/v4/search?q={search_term}&token=9711f50cd8d81cc3f4406fb08839b378"
		response = requests.get(url)
		news = response.json()
		return news
		
	def get_fin_statements(self, symbol):
		df = Ticker(symbol)
		df1 = df.income_statement().reset_index(drop = True).transpose()
		df2 = df.balance_sheet().reset_index(drop = True).transpose()
		df3 = df.cash_flow().reset_index(drop = True).transpose()
		return df1, df2, df3
		
	def fundamental(self, query):
		symbol = query
		articles_string = ""
		financial_statement = ""
		fundamental_data = ""
	
		gnews_api_spec = self.get_gnews_api_spec(symbol)
		gnews_api = self.get_gnews_api()
		
		date_now = datetime.now()
		date_year = date_now.year
		date_month = date_now.month
		date_day = date_now.day

		date_d = "{}-{}-{}".format(date_year, date_month, date_day)
		
		#try:
		income_statement, balance_sheet, cash_flow = self.get_fin_statements(symbol)
		financial_statement += (f"{symbol}") + "\n\n"
		#df_price = self.get_stock_history(symbol, date_d)
		#stock_price = self.get_latest_price(df_price)
		#financial_statement += "Current Market Price:" + "\n\n"
		#financial_statement += (f"{stock_price}") + "\n\n"
		financial_statement += "Income Statement:" + "\n\n"
		financial_statement += income_statement.to_string() + "\n\n"
		financial_statement += "Balance Sheet:" + "\n\n"
		financial_statement += balance_sheet.to_string() + "\n\n"
		financial_statement += "Cash Flow:" + "\n\n"
		financial_statement += cash_flow.to_string() + "\n\n"
		#except:
			#print(f"Financial Statement for _{symbol}_ couldn't be found 😥️")
			
		for article in gnews_api_spec['articles']:
			article_string = f"**Title:** {article['title']}, **Description:** {article['description']} \n"
			articles_string += article_string + "\n"
			
		for article in gnews_api['articles']:
			article_string_ = f"**Title:** {article['title']}, **Description:** {article['description']} \n"
			articles_string += article_string_ + "\n"
			
		fundamental_data = "Documented Knowledge:" + "\n\n"
		fundamental_data += articles_string + "\n\n"
		fundamental_data += "Financial Statements" + "\n\n"
		fundamental_data += financial_statement
	
		return fundamental_data
