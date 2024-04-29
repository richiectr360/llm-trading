from langchain.agents import AgentType, initialize_agent, load_tools, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import ChatMessage, AgentAction, AgentFinish, OutputParserException
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import StringPromptTemplate
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.preprocessing import MinMaxScaler
from langchain.chains.llm import LLMChain
from langchain.llms import LlamaCpp
import plotly.graph_objects as go
from langchain.llms import OpenAI
from torchview import draw_graph
from tiingo import TiingoClient
from typing import List, Union
from datetime import timedelta
from datetime import datetime
from pandas import Timestamp
from Analysis import Analysis
from Forecasts import Forecasts
from PIL import Image
import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import asyncio
import base64
import time
import json
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

st.set_page_config(page_title = "DNN Lab", page_icon = "🌌️", layout = "wide", initial_sidebar_state = "expanded")
container = st.container()
img = Image.open("./Backgrounds/White Template Logo.png")
img = img.resize((300, 75))
container.image(img)
agree = st.checkbox('Show Model')

openai_api_key = "<API_KEY>"

class Background:
	def __init__(self, img):
		self.img = img
	def set_back_side(self):
		side_bg_ext = 'png'
		side_bg = self.img

		st.markdown(
		f"""
		<style>
			[data-testid="stSidebar"] > div:first-child {{
				background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
			}}
		</style>
		""",
		unsafe_allow_html = True,
		)
		
class CustomPromptTemplate(StringPromptTemplate):
	template: str
	tools: List[Tool]

	def format(self, **kwargs) -> str:
		intermediate_steps = kwargs.pop("intermediate_steps")
		thoughts = ""
		for action, observation in intermediate_steps:
			thoughts += action.log
			thoughts += f"\nObservation: {observation}\nThought: "
		kwargs["agent_scratchpad"] = thoughts
		kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
		kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
		return self.template.format(**kwargs)
		
class CustomOutputParser(AgentOutputParser):
	def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
		if "Final Answer:" in llm_output:
			return AgentFinish(
				return_values = {"output": llm_output.split("Final Answer:")[-1].strip()},
				log = llm_output,
			)
		regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
		match = re.search(regex, llm_output, re.DOTALL)
		if not match:
			raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
		action = match.group(1).strip()
		action_input = match.group(2)
		return AgentAction(tool = action, tool_input = action_input.strip(" ").strip('"'), log = llm_output)
		
class ReinforcementLab:
	def __init__(self, vis_graph = True):
		self.grp_vis = vis_graph
		self.back = Background('./Backgrounds/augmented_bulb_2.png')
		self.back.set_back_side()
		self.llm = OpenAI(model_name = "gpt-4-1106-preview", temperature = 0, streaming = True, api_key = openai_api_key)
		self.tools = load_tools(["ddg-search"])
		self.agent = initialize_agent(self.tools, self.llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = False)
		self.prompt_template_search = ""
		self.fundamental_query = ""
		self.template_fundamental = ""
		self.anali = Analysis()
		self.tools_custom = [
			Tool(
				name = "Fundamental_Analysis",
				func = self.anali.fundamental,
				description = "Necesary for when in need of making Fundamental Analyisis"
			)
		]
		self.output_parser = CustomOutputParser()
		self.tool_names = [tool.name for tool in self.tools_custom]
		self.memory = ConversationBufferWindowMemory(k = 2)
		self.device = "cuda"
		self.config = {}
		self.client = None
		self.startDate = dt.datetime(1997, 9, 22)
		self.endDate = dt.datetime(1997, 9, 22)
		if 'comp_prompt' not in st.session_state:
			st.session_state.comp_prompt = {}
		if 'symbol' not in st.session_state:
			st.session_state.symbol = {}
		if 'fetch_data' not in st.session_state:
			st.session_state.fetch_data = {}
		if 'fundamental_analysis' not in st.session_state:
			st.session_state.fundamental_analysis = {}
		if 'trained_model' not in st.session_state:
			st.session_state.trained_model = {}
			
	async def get_stock_symbol(self, company_name):
		st_callback = StreamlitCallbackHandler(st.container())
		search_results = self.agent.run(company_name + self.prompt_template_search, callbacks = [st_callback])
		symbols = search_results.split(" ")
		return symbols
		
	async def get_stock_history(self, symbol, date):
		ticker = yf.Ticker(symbol)
		data = ticker.history(start = "2019-01-01", end = date)
		return data
		
	async def fetch_data(self, df, symbol):
		data_date = df.index.to_numpy().reshape(1, -1)
		data_open_price = df['Open'].to_numpy().reshape(1, -1)
		data_high_price = df['High'].to_numpy().reshape(1, -1)
		data_low_price = df['Low'].to_numpy().reshape(1, -1)
		data_close_price = df['Close'].to_numpy().reshape(1, -1)
		df_datas = np.concatenate((data_date, data_open_price, data_high_price, data_low_price, data_close_price), axis = 0)
		return df_datas
		
	async def fetch_data_daily(self, df, symbol):
		df.index = df.index - timedelta(hours = 6)
		data_date = df.index.to_numpy().reshape(1, -1)
		data_open_price = df['open'].to_numpy().reshape(1, -1)
		data_high_price = df['high'].to_numpy().reshape(1, -1)
		data_low_price = df['low'].to_numpy().reshape(1, -1)
		data_close_price = df['close'].to_numpy().reshape(1, -1)
		df_datas = np.concatenate((data_date, data_open_price, data_high_price, data_low_price, data_close_price), axis = 0)
		return df_datas
		
	async def get_stock_price(self, symbol):
		ticker = yf.Ticker(symbol)
		data = ticker.history(period = "1d", interval = "1m")
		return data
		
	async def get_latest_price(self, data):
		latest_price = []
		if not data.empty:
			latest_price.append(data['Close'].iloc[-1])
			latest_prices = np.array(latest_price).reshape(-1, 1)
			return latest_prices
		else:
			return None
			
	async def encode_text_with_gpt2(self, text, tokenizer, model, device):
		tokenizer.pad_token = tokenizer.eos_token

		tokens = tokenizer.encode_plus(text,
						add_special_tokens = True,
						max_length = tokenizer.model_max_length,
						truncation = True,
						padding = 'max_length',
						return_tensors = 'pt')

		tokens = {key: val.to(device) for key, val in tokens.items()}

		with torch.no_grad():
			outputs = model(**tokens)
			embeddings = outputs.last_hidden_state

		input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
		sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1e-9)
		mean_embeddings = sum_embeddings / sum_mask

		return mean_embeddings
			
	async def update_graph(self, symbol, fundamental_analysis, predicted_day, plot_placeholder, price_placeholder, update_interval = 60):
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		model = GPT2Model.from_pretrained('gpt2')
		model.to(self.device)
		fig = go.Figure()
		fig.add_trace(go.Scatter(x = [], y = [], name = 'Price', line = dict(color = "#fd7f20")))
		
		history = self.client.get_dataframe(symbol,
							startDate = self.startDate,
							endDate = self.endDate,
							frequency = '1Min')
							
		stock_data = await self.fetch_data_daily(history, symbol)
		data_date_daily = stock_data[0]
		data_close_price = stock_data[4]
		
		fig.data[0].x = data_date_daily
		fig.data[0].y = data_close_price
		
		scaler_daily = MinMaxScaler(feature_range = (0, 1))
		normalized_daily = scaler_daily.fit_transform(np.array(data_close_price).reshape(-1, 1)).reshape(1, 1, -1)
		
		embeddings = await self.encode_text_with_gpt2(fundamental_analysis, tokenizer, model, self.device)
		embeddings = embeddings.cpu().detach().numpy().reshape(1, 1, -1)
		
		price_dump = np.array([1]).reshape(1, 1, -1)
		normalized_daily = np.concatenate((normalized_daily, price_dump), axis = 2)
		start_time = time.time()
		
		while True:
			data = await self.get_stock_price(symbol)
			prices = await self.get_latest_price(data)
			price = prices[0][0]
			
			if price is not None:
				price_print = str("$" + str(round(price, 2)) + " USD")
				new_timestamp = Timestamp(datetime.now())
				data_date_daily = np.append(data_date_daily, new_timestamp)
				data_close_price = np.append(data_close_price, price)
				
				fig.data[0].x = data_date_daily
				fig.data[0].y = data_close_price
				
				fig.update_layout(title_text = "Data to the second")
				plot_placeholder.plotly_chart(fig)
				
				price_placeholder.header(f" :green[_{price_print}_]")
				
				price_ = scaler_daily.transform(np.array(price).reshape(-1, 1))[0][0]
				normalized_daily[0][0][-1] = price_
				elapsed_time = time.time() - start_time
				
				if elapsed_time > update_interval:
					start_time = time.time()
					price_dump = np.array([price_]).reshape(1, 1, -1)
					normalized_daily = np.concatenate((normalized_daily, price_dump), axis = 2)
				
			await asyncio.sleep(1)
		
	async def run(self):
		self.config['session'] = True
		self.config['api_key'] = "<API_KEY>"
		self.client = TiingoClient(self.config)
		with open('./prompts.json', 'r') as f:
			st.session_state.comp_prompt = json.load(f)
		self.prompt_template_search = st.session_state.comp_prompt["prompts"][0]["search_comp_prompt"]
		date_now = datetime.now()
		date_year = date_now.year
		date_month = date_now.month
		date_day = date_now.day
		date_day_ = date_now.strftime("%A")
		
		self.startDate = dt.datetime(date_year, date_month, date_day)
		self.endDate = dt.datetime(date_year, date_month, date_day)
		
		date_d = "{}-{}-{}".format(date_year, date_month, date_day)
		
		st.title(":orange[Welcome!]")
		st.subheader(f" _{date_d}_")
		st.subheader(f":orange[{date_day_}]", divider = 'rainbow')
		
		with st.form(key = 'company_search_form'):
			company_name = st.text_input("Enter a company name:")
			submit_button = st.form_submit_button("Search", type = "primary")
			
		if company_name:
			stock_price = None
			if company_name in st.session_state.symbol:
				symbols = st.session_state.symbol[company_name]
			else:
				symbols = await self.get_stock_symbol(company_name)
				st.session_state.symbol[company_name] = symbols
				
			tasks = []
				
			for symbol in symbols:

				left_column, right_column = st.columns(2)

				with left_column:
					st.header(symbol)
					fundamental_analysis_placeholder = st.container()
					live_plot_placeholder = st.empty()
				with right_column:
					price_placeholder = st.empty()
					plot_placeholder = st.empty()
				with left_column:
					plot_placeholder_daily = st.empty()
					price_pred_placeholder = st.empty()
				with right_column:
					table_placeholder_daily = st.empty()

				st.markdown("""---""")
				
				if symbol in st.session_state.fetch_data:
					stock_data = await self.fetch_data(st.session_state.fetch_data[symbol], symbol)
				else:
					df = await self.get_stock_history(symbol, date_d)
					stock_data = await self.fetch_data(df, symbol)
					st.session_state.fetch_data[symbol] = df
					
				while stock_price == None:
					data = await self.get_stock_price(symbol)
					stock_price = await self.get_latest_price(data)
					
				num_data_points = len(stock_data[0])
				
				feg = go.Figure(data = [go.Candlestick(x = stock_data[0],
									open = stock_data[1],
									high = stock_data[2],
									low = stock_data[3],
									close = stock_data[4])])
				feg.update_layout(title_text = "Full Data")
				feg.update_layout(xaxis_rangeslider_visible = False)
				plot_placeholder.plotly_chart(feg, use_container_width = True)
				
				self.fundamental_query = st.session_state.comp_prompt["prompts"][0]["fundamental_analysis_prompt_template"]
				self.fundamental_query += symbol + " Latest price for this company is of: " + str(stock_price)
				self.template_fundamental = st.session_state.comp_prompt["prompts"][0]["prompt_fundamental_roll"]
				
				prompt_fund = CustomPromptTemplate(template = self.template_fundamental, tools = self.tools_custom, input_variables = ["input", "intermediate_steps", "history"])
				llm_chain = LLMChain(llm = self.llm, prompt = prompt_fund)
				agent_anali = LLMSingleActionAgent(llm_chain = llm_chain, output_parser = self.output_parser, stop = ["\nObservation:"], allowed_tools = self.tool_names)
				anali_agent_executor = AgentExecutor.from_agent_and_tools(agent = agent_anali, tools = self.tools_custom, verbos = True, memory = self.memory)
				
				if symbol in st.session_state.fundamental_analysis:
					fundamental_analysis = st.session_state.fundamental_analysis[symbol]
				else:
					st_callback = StreamlitCallbackHandler(fundamental_analysis_placeholder)
					fundamental_analysis = anali_agent_executor.run(self.fundamental_query, callbacks = [st_callback])
					st.session_state.fundamental_analysis[symbol] = fundamental_analysis
					
				fore = Forecasts(stock_data)
				model, dataset_train, dataset_val, unseen_data, scaler, split_index = fore.prepare_model()
				
				if self.grp_vis:
					with st.sidebar:
						with st.expander(symbol):
							model_graph = draw_graph(model,
										input_size = dataset_train.x.shape,
										expand_nested = True)
							model_graph.visual_graph
							model_graph.resize_graph(scale = 5.0) # scale as per the view 
							model_graph.visual_graph.render(format = 'svg')
							
				if symbol in st.session_state.trained_model:
					trained_model = st.session_state.trained_model[symbol]
				else:
					my_bar = st.progress(0, text = "Strating...")
					trained_model = fore.train_forecast(model, dataset_train, dataset_val, my_bar)
					st.session_state.trained_model[symbol] = trained_model
					
				fag, fog, FinalPred, predicted_day = fore.launch_forecast(trained_model, num_data_points, dataset_train, dataset_val, unseen_data, scaler, split_index, go)
				table_placeholder_daily.plotly_chart(fag, use_container_width = True)
				plot_placeholder_daily.plotly_chart(fog, use_container_width = True)
				price_pred_placeholder.dataframe(FinalPred)
				
				task = asyncio.create_task(self.update_graph(symbol, fundamental_analysis, predicted_day, live_plot_placeholder, price_placeholder))
				tasks.append(task)
			await asyncio.gather(*tasks)
		
if __name__ == "__main__":
	reinf = ReinforcementLab(vis_graph = agree)
	asyncio.run(reinf.run())
