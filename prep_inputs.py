import sys
import importlib
import pandas as pd
import numpy as np
import math
# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

from datetime import date
from datetime import datetime
import pdb
import comp_data
from yfinance import yfinance as yf
from yahoo_fin.yahoo_fin import stock_info as si

import ipywidgets as widgets
from IPython.display import display, clear_output

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials
import gspread_dataframe
from gspread_dataframe import get_as_dataframe, set_with_dataframe

from google.auth import default
creds, _ = default()

#gc = gspread.authorize(GoogleCredentials.get_application_default())
gc = gspread.authorize(creds)

from fin_gists import damoCF as dacf
from fin_gists import shared_vars as sv

style = {'description_width': 'initial'}
marketdata = comp_data.Market()

import asyncio

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator

def read_sheet(sheet):
  df = get_as_dataframe(sheet)
  df.replace(np.nan, '', regex=True,inplace=True)
  df.filter(regex='^((?!Unnamed).)*$',axis=1)
  return df.filter(regex='^((?!Unnamed).)*$',axis=1)
  
def read_DB(gc,filn='StockDB'):
  ## This function can be made more elegant and streamlined to automatically identify the sheets and download in dfs/dict
  workbook = gc.open(filn)
  sheets = workbook.worksheets()
  #sheets = ["Ticker",'Lease','Optionholdings']
  DBdict = {isheet.title: read_sheet(isheet) for isheet in sheets}
  return DBdict

def export_to_sheets(gc,df,workbook="StockDB",worksheet_name='Optionholdings',mode='r'):
    ws = gc.open(workbook).worksheet(worksheet_name)
    if(mode=='w'):
        ws.clear()
        set_with_dataframe(worksheet=ws,dataframe=df,include_index=False,include_column_header=True,resize=True)
        return True
    elif(mode=='a'):
        ws.add_rows(df.shape[0])
        set_with_dataframe(worksheet=ws,dataframe=df,include_index=False,include_column_header=False,row=ws.row_count+1,resize=False)
        return True
    else:
        return get_as_dataframe(worksheet=ws)

def export_to_sheets_gs(gc,df,workbook="StockDB",worksheet_name='Optionholdings'):
  ws = gc.open(workbook).worksheet(worksheet_name)
  params = {'valueInputOption': 'USER_ENTERED'}
  body = {'values': df.values.tolist()}
  ws.values_append(f'{sheet_name:str}!A1:G1', params, body)
  return True

def write_rowDB(gc,dfrow,sheetn='Optionholdings',filn='StockDB'):
  workbook = gc.open(filn)
  gcsheet = workbook.worksheet(sheetn)
  gcsheet.append_row(dfrow)

def get_yahoo_fin(Ticker='MSFT'):
  ## get data from the yahoo_fin table - some data elements are not available in the yfinance app
  si_dict = si.get_financials(Ticker, yearly = True, quarterly = True)
  for i,(k,v) in enumerate(si_dict.items()):
    si_dict[k] = v.astype(float)
  inc_stat = si_dict['yearly_income_statement']  
  qbal_sheeet = si_dict['quarterly_balance_sheet']
  bal_sheeet = si_dict['yearly_balance_sheet']
  ## EBITDA, DA get the 
  revenue = inc_stat.loc('totalRevenue')
  grossprofit = inc_stat.loc('grossProfit')
  cagr = revenue[1:-1]/revenue[2:]-1 # skip the ttm and breakdown column 
  #ebitda = inc_stat.loc('normalizedEBITDA') ## ebit and ebitda are not correct in this scrappers..
  #ebitda_margin = ebitda[1:]/revenue[1:]
  #da = inc_stat.loc('Reconciled Depreciation')
  #ebit_margin = ebitda_margin + da[1:]/revenue[1:]

  #dilutedshares = bal_sheet.loc['sharesIssued'] - turns out the new yahoo_fin doesnt have this anyway..
  #dilutionrate = dilutedshares[:-1]/dilutedshares[1:]-1 

  y_dict = locals()
  return y_dict

    
def comp_finpop(comp):
  #y_dict = get_yahoo_fin(comp.ticksym)
  quarterly_financials = comp.quarterly_financials.fillna(0)
  quarterly_balance_sheet = comp.quarterly_balance_sheet.fillna(0)
  financials = comp.financials.fillna(0)
  cashflow = comp.cashflow.fillna(0)

  comp.ttm_revs = sum(quarterly_financials.loc['Total Revenue']) #
  comp.ttm_ebit = sum(quarterly_financials.loc['Ebit'])
  comp.ttm_gross_profit = sum(quarterly_financials.loc['Gross Profit'])
  comp.ttm_gross_profit_margin = comp.ttm_gross_profit/comp.ttm_revs
  comp.gross_profit_margin = np.mean(financials.loc['Gross Profit']/financials.loc['Total Revenue'])
  try:
    shortinv = quarterly_balance_sheet.loc['Short Term Investments'].iloc[0]
  except: 
    shortinv =0
    print('** There seem to be no short term investments or marketable securities - perhaps already clubbed in Cash?')
  comp.cash_mms = quarterly_balance_sheet.loc['Cash'].iloc[0]+shortinv
  comp.short_longterm_debt = 0 if 'Short Long Term Debt' not in quarterly_balance_sheet.index else quarterly_balance_sheet.loc['Short Long Term Debt'].iloc[0]
  comp.longterm_debt = 0
  try:
    comp.longterm_debt = quarterly_balance_sheet.loc['Long Term Debt'].iloc[0] 
  except:
    print('There is no Long term debt! or it is corrupted!')
  comp.net_debt = comp.short_longterm_debt + comp.longterm_debt - comp.cash_mms
  try:
    interest_expense = sum(quarterly_financials.loc['Interest Expense'])
  except:
    interest_expense = 0
  comp.interest_expense = interest_expense/comp.net_debt
  
  ## dividends
  try:
    comp.divhist = -cashflow.loc['Dividends Paid'].to_numpy()/comp.info['sharesOutstanding']  #y_dict['dilutedshares']
    comp.dividendgrowth = comp.divhist[:-1]/comp.divhist[1:]-1
    comp.avgdivgrowth = np.mean(comp.dividendgrowth)
  except: 
    comp.avgdivgrowth = 0
  
  # ebit, ebitda, tax, interest, da The ebit, operating income, ebitda etc in the yfinance are wrong
  comp.revenue = financials.loc['Total Revenue']
  comp.ebit = financials.loc['Net Income Applicable To Common Shares']+financials.loc['Income Tax Expense']-financials.loc['Interest Expense']
  comp.ttm_ebit = sum(quarterly_financials.loc['Net Income Applicable To Common Shares']+ 
                      quarterly_financials.loc['Income Tax Expense']- quarterly_financials.loc['Interest Expense'])
  comp.tax_rate = np.mean(financials.loc['Income Tax Expense']/comp.ebit) # avg over past few years
  comp.interest_rev = np.mean(financials.loc['Interest Expense']/comp.revenue) # avg as a % OF REVENUE over past few years
  comp.da = cashflow.loc['Depreciation']
  comp.ebitda = comp.ebit + comp.da
  comp.ebit_margin = comp.ebit/comp.revenue
  comp.ebitda_margin = comp.ebitda/comp.revenue
  comp.da_rate = comp.da/comp.revenue
  comp.mean_ebitda_margin = np.mean(comp.ebitda_margin)
  comp.mean_ebit_margin = np.mean(comp.ebit_margin)
  comp.mean_da_rate = np.mean(comp.da_rate)
  comp.mean_interest_rev = np.mean(comp.interest_rev) # of revnenue
  #comp.tax_rate = np.mean(financials.loc['Income Tax Expense']/financials.loc['Ebit']) # avg over past few years
  comp.rnd = financials.loc['Research Development']
  comp.rnd_dict = dacf.rnd_conv(comp)
  comp.curr_cagr = dacf.get_cagr(comp)
  comp.marketdata = comp_data.Market()
  comp.equity_book_value = comp.quarterly_balance_sheet.loc['Total Stockholder Equity'].iloc[0]
  comp.sales2cap_approx = comp.ttm_revs/(comp.equity_book_value+comp.net_debt) # actual capital will need debt tments for lease, rnd etc
  
  return comp
 
def get_ticker(DBdict):
  ## Enter your Ticker Symbol - make sure no mistakes!!
  ## If this symbol exists in my DB - I am going to get its latest data else empty df
  ## populate the cells - either from spreadhseet or calculations or defaults
  ## Where the spreadsheet populates but we have more recent data - print the new data to allow user to decide
  global dfts, dfls, dfos, comp, rnd_dict
  
  dft = DBdict['Ticker']
  dfl = DBdict['Lease']
  dfo = DBdict['Optionholdings']
  
  ticksym = widgets.Text(
      value='MSFT',
      placeholder='Type something',
      description='Ticker Symbol:',
      disabled=False,
      style=style,
      continuous_update = False
  )
  
  def f(ticksym):
    ## If this symbol exists in my DB - I am going to get its latest data
    global dfts, dfls, dfos, comp, rnd_dict
    comp = sv.comp

    dftickset = dft[dft['Ticker']==ticksym.upper()]
    if len(dftickset):
      dfts = dftickset.iloc[-1]
      dfls = dfl[dfl['UUID'].astype(str)==dfts['UUID']]
      dfos = dfo[dfo['UUID'].astype(str)==dfts['UUID']]
      print(ticksym, ' Last Updated on ', dfts['LastUpdate'])
    else:
      dfts = dft.loc[0].transpose()  #.iloc[[0]].copy() #default take the first row
      dfls = dfl.loc[0].transpose()  #.iloc[[0]].copy()
      dfos = dfo.loc[0].transpose()  #.iloc[[0]].copy()
      print(ticksym, ' NOT FOUND in DB - Using defaults please update appropriately')
    # get comp info
    #print(dfts)
    comp = yf.Ticker(ticksym)
    comp.ticksym = ticksym.upper()
    comp = comp_finpop(comp)

    sv.comp = comp
    sv.Inp_dict['rnd_dict'] = comp.rnd_dict
    sv.Inp_dict['Ticker'] = ticksym
    #get_lease_opt()

  ui =  {'ticksym': ticksym}
  out = widgets.interactive_output(f,ui)
  display(ticksym,out)
  tick_dict = {'title':'Symbol','ui': ticksym,'out':out}
  return #tick_dict
  
def get_lease_opt():
  ## get the lease input and options data
  ## Make a widgets and prepopulate them? Allow user to modify if need be
  ## 
  global dfls, dfos
  lslist = [widgets.FloatText(description = 'Year' + str(i), value = dfls['Year'+str(i)],readout_format=':,.2f') for i in range(6)]
  lstup = ['Year' + str(i) for i in range(6)]
  lsdict = dict(zip(lstup,lslist))
  lsdict['bulk_commitment'] = widgets.FloatText(description = 'bulk_commitment', value = dfls['bulk_commitment'],style=style)
  lsdict['nyrs_bulk'] = widgets.FloatText(description = 'N yrs bulk', value = dfls['nyrs_bulk'],style=style)
  lsdict['cost_of_debt'] = widgets.FloatText(description = 'debt cost', value = dfls['cost_of_debt'],style=style)

  #globals().update(lsdict)
  lease_inp = {}
  lease_inp['Year'] = np.arange(1,6)
  def flease(**lsdict):
    lease_inp['current_commitment'] = lsdict[lstup[0]]*1e6  ## original entries are in $M
    lease_inp['future_commitments'] = [lsdict[lstup[i]]*1e6 for i in range(1,6)]
    lease_inp['bulk_commitment'] = lsdict['bulk_commitment']*1e6
    lease_inp['nyrs_bulk'] = lsdict['nyrs_bulk']
    lease_inp['cost_of_debt'] = lsdict['cost_of_debt']
    lease_dict = dacf.lease_conv(lease_inp)
    comp.ebit_adj = comp.ttm_ebit + lease_dict['adj_op_income'] + comp.rnd_dict['adj_op_income']
    comp.mean_margin = comp.ebit_adj/comp.ttm_revs
    sv.Inp_dict['lease_dict'] = lease_dict
    sv.comp = comp
    for k,v in lsdict.items(): sv.Inp_dict[k] = v
    #get_options()
  
  layout =widgets.Layout(grid_template_columns='1fr 1fr 1fr')
  lsui = widgets.GridBox( tuple(lsdict.values()),layout = layout)
  lsout = widgets.interactive_output(flease, lsdict)
  ltit = widgets.HTML('<h4> Lease Commitment Inputs ($M) </h4>')
  display(ltit)
  display(lsui, lsout)

  ## Options inputs
  opdict = {} 
  opt_dict = {}
  opdict['strike'] = widgets.FloatText(description = 'Avg Strike', value = dfos['strike'],style=style)
  opdict['expiration'] = widgets.FloatText(description = 'Avg Expiration', value = dfos['expiration'],style=style)
  opdict['n_options'] = widgets.FloatText(description = 'Num of Options', value = dfos['n_options'],style=style)
  def foptions(**opdict):
    opt_dict['strike'] = opdict['strike']
    opt_dict['expiration'] = opdict['expiration']
    opt_dict['n_options'] = opdict['n_options']
    sv.comp.opt_dict = opt_dict
    for k,v in opdict.items(): sv.Inp_dict[k] = v

  opui = widgets.GridBox(tuple(opdict.values()),layout = layout)
  opout = widgets.interactive_output(foptions, opdict)
  otit = widgets.HTML('<h4> Options Outstanding Inputs </h4>')
  display(otit)
  display(opui, opout)
  options_ui_dict = {'title':otit,'ui':opui,'out':opout}
  lease_ui_dict = {'title':ltit,'ui':lsui,'out':lsout}
  return #lease_ui_dict, options_ui_dict

def get_options():
  ## Options inputs
  global dfos
  opdict = {}
  opt_dict = {} # we may have some duplication here.
  opdict['strike'] = widgets.FloatText(description = 'Avg Strike', value = dfos['strike'],style=style)
  opdict['expiration'] = widgets.FloatText(description = 'Avg Expiration', value = dfos['expiration'],style=style)
  opdict['n_options'] = widgets.FloatText(description = 'Num of Options', value = dfos['n_options'],style=style)
  def foptions(**opdict):
    opt_dict['strike'] = opdict['strike']
    opt_dict['expiration'] = opdict['expiration']
    opt_dict['n_options'] = opdict['n_options']
    sv.comp.opt_dict = opt_dict
    for k,v in opdict.items(): 
      sv.Inp_dict[k] = v
      dfos.loc[0,k] = v
    #value_inputs()

  layout =widgets.Layout(grid_template_columns='1fr 1fr 1fr')
  opui = widgets.GridBox(tuple(opdict.values()),layout = layout)
  opout = widgets.interactive_output(foptions, opdict)
  otit = widgets.HTML('<h4> Options Outstanding Inputs </h4>')
  display(otit)
  display(opui, opout)
  options_ui_dict = {'title':otit,'ui':opui,'out':opout}
  return 

def value_inputs():
  country_name_list = list(marketdata.get_country_tax_rates().index) + ['']
  industry_name_list = comp_data.industry_name_list + ['']
  dfts_list = dfts.index
  #dfts_res = [i for i in dfts_list if 'Unnamed' not in i] 
  lsdts_int = ['year_conv',	'terminal_year']
  lsdts_flt1a = ['beg_cagr',	'long_term_cagr','beg_margin']
  lsdts_flt1b = ['long_term_margin',	'tax_rate','wacc','long_term_coc']
  lsdts_flt2 = ['sales_to_capital', 'minority_interest','crossholdings_nonopassets' ]
  lsdts_indt =['Industry1',	'Industry2', 'Industry3']
  lsdts_indf = ['Industry1Wt',	'Industry2Wt', 'Industry3Wt']
  lsdts_cont = ['Country1','Country2','Country3']
  lsdts_conf = ['Country1Wt','Country2Wt','Country3Wt']
  lsdts_liqp = ['prob_failure','distress_price']
  lsdts_liqt = ['liquidation_type']
  lsdts_flt3 = ['NOLbase']
  lsdts_txt1 = ['Story/Rationale']
  lsdts_reit = ['darate','long_da_rate','sbc','beg_int','long_term_int','stock_dilution_rate','maintcapex','long_tax_rate','beg_div_growth','long_div_growth','cap_rate']

  #pdb.set_trace()
  #print(industry_name_list)
  #print(lsdts_indt)
  #print(dfts)

  dfts_dict = {i: widgets.IntText(description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_int }
  #dfts_dict.update({'Forecast': widgets.HTML('<b>Time Horizon</b>')})
  #dfts_dict.update({'GrowthMargins': widgets.HTML('<b>Growth & Margins</b>')})
  dfts_dict.update({i: widgets.FloatSlider(description=i,min=-0.5,max=1,step=0.01,value=dfts[i],style=style,continuous_update=False,readout_format='.0%') for i in lsdts_flt1a})
  dfts_dict.update({i: widgets.FloatSlider(description=i,min=0,max=1,step=0.01,value=dfts[i],style=style,continuous_update=False,readout_format='.0%') for i in lsdts_flt1b})
  dfts_dict.update({i: widgets.FloatText(description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_flt2})
  dfts_dict.update({i: widgets.Dropdown(options=industry_name_list, description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_indt})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1,step=0.05, description=i,value=dfts[i],style=style,continuous_update=False,readout_format='.0%') for i in lsdts_indf})
  dfts_dict.update({i: widgets.Dropdown(options=country_name_list, description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_cont})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1,step=0.05, description=i,value=dfts[i],style=style,continuous_update=False,readout_format='.0%') for i in lsdts_conf})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1,step=0.05, description=i,value=dfts[i],style=style,continuous_update=False,readout_format='.0%') for i in lsdts_liqp})
  dfts_dict.update({i: widgets.Dropdown(options=[('Fair Value', 'V'), ('Book Value', 'B')], description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_liqt})
  dfts_dict.update({i: widgets.FloatText(description=i,value=dfts[i],style=style,continuous_update=False) for i in lsdts_flt3})
  dfts_dict.update({i: widgets.FloatSlider(description=i,min=0,max=1,step=0.001,value=dfts[i],style=style,continuous_update=False,readout_format='.1%') for i in lsdts_reit})

  story_dict = {i: widgets.Textarea(description=i,value=dfts[i],style=style,continuous_update=False,layout={'height':'100%','width':'1000px'}) for i in lsdts_txt1}
  #@debounce(0.5)
  def fstory(**story_dict):
    for k,v in story_dict.items():
      sv.Inp_dict[k] = v
  story_out = widgets.interactive_output(fstory, story_dict)
  story_layout =widgets.Layout(grid_template_columns='1fr',width='1000px')
  story_ui = widgets.Box(tuple(story_dict.values()),layout=story_layout)
  display(story_ui,story_out)

  #out_gen = widgets.Output(layout={'border': '1px solid black'},wait=True)
  #display(out_gen)
  
  ind_df = pd.DataFrame()
  #@debounce(.5)
  def finpdict(**dfts_dict):
    comp = sv.comp
    for k,v in dfts_dict.items():
      sv.Inp_dict[k] = v
    
    value_op_outstanding = dacf.option_conv(comp)
    sv.Inp_dict['value_op_outstanding'] = value_op_outstanding
    comp.long_tax_rate, comp.df_country_tax = dacf.get_market_info(sv.Inp_dict,metric='long_tax_rate')
    comp.long_term_coc, comp.df_ind_coc = dacf.get_industry_info(sv.Inp_dict,metric='long_term_coc')
    comp.ind_beta, comp.df_ind_beta = dacf.get_industry_info(sv.Inp_dict,metric='beta')
    comp.wacc = dacf.get_wacc(sv.comp)
    
    indt_list_tmp = [v for k,v in sv.Inp_dict.items() if k in lsdts_indt]
    indt_list = list(filter(None, indt_list_tmp))  ## remove empty ones
    print('# of Ind ',len(indt_list))
    for iindt in indt_list:
      if iindt not in ind_df.columns.tolist(): #only do this for new industry value
        ind_dat = comp_data.Industry(iindt)
        tmp_df = ind_dat.get_historical_growth()
        tmp_df = tmp_df.append(ind_dat.get_margins()[['net margin','pre-tax unadjusted operating margin','pre-tax lease & r&d adj margin']])
        tmp_df.loc['sales/capital'] = ind_dat.get_capital_expenditures()['sales/ invested capital']
        tmp_df = tmp_df.append(ind_dat.get_industry_tax_rates()[['average across only money-making companies2','aggregate tax rate3']])
        tmp_df = tmp_df.append(comp.df_ind_coc[iindt])
        tmp_df = tmp_df.append(comp.df_ind_beta[iindt])
        ind_df[iindt] = tmp_df
      
    ## Relevant Metrics from Company's recent financials
    display(widgets.HTML('<h4> Metrics from Company Recent Financials & some basic calcs </h4>'))
    listvar = ['ebit_adj','ttm_ebit','mean_margin','curr_cagr',
               'interest_expense','wacc','long_term_coc','ind_beta','sales2cap_approx',
               'tax_rate','long_tax_rate','avgdivgrowth','mean_da_rate','mean_interest_rev','mean_ebitda_margin','mean_ebit_margin']
    list_dict = {i:eval("comp."+i) for i in listvar}
    listformat = [' {:,.0f}']*2 + [' {:.1%}']*(len(listvar)-2)
    dictformat = dict(zip(list_dict.keys(),listformat))
    #print(list_dict)
    display(pd.DataFrame(data=list_dict.values(),
                       columns=[comp.ticksym],index=list_dict.keys())
                       .transpose().style.format(dictformat))

    '''## display few yahoo_finance metrics
    display(widgets.HTML('<h4> Additional date from yahoo_finance & some basic calcs </h4>'))
    ylist = ['ebitda_margin','ebit_margin','dilutionrate']
    ylistform = ['{:.1%}']*(len(ylist))
    ydictformat = zip(ylist,ylistform)
    display(pd.DataFrame(data = comp.y_dict[ylst].style.format(ydictformat)))
    '''
    
    ## Relevant Industry Metrics
    display(widgets.HTML('<h4> Key Industry Metrics - Use as Reference </h4>'))
    print(ind_df[indt_list])
    
    ## Relevant Country of operation Metrics
    #prev_year = str(int(datetime.today().strftime('%Y'))-1)
    prev_year = marketdata.get_country_tax_rates().columns.max()
    display(widgets.HTML(value='<h4> Key Country Level Metrics from ' + prev_year + ' - Use as Reference </h4>'))
    cont_dict = {v: marketdata.get_country_tax_rates().loc[v,prev_year] for k,v in sv.Inp_dict.items() if k in lsdts_cont}
    cont_df = pd.DataFrame(cont_dict,index=['Tax rates'])
    print(cont_df)
    #print(comp.dfcountry_tax)
    
    sv.comp = comp
    
  layout =widgets.Layout(grid_template_columns='1fr 1fr 1fr')
  dfts_ui = widgets.GridBox( tuple(dfts_dict.values()),layout = layout)
  dfts_out = widgets.interactive_output(finpdict, dfts_dict)
  inptit = widgets.HTML('<h2> Key Value Inputs </h2>')
  display(inptit)
  display(dfts_ui, dfts_out)
  value_dict = {'title':inptit,'ui':dfts_ui,'out':dfts_out}
  return value_dict  #, out_gen

def append_todb(gc,df,work_sheet):
  ndf = df.copy() if isinstance(df,pd.DataFrame) else pd.DataFrame(df).transpose()
  for i in ndf.columns: ndf[i] = sv.Inp_dict[i] ## assumes all columns are in Inp_dict
  ## now write to BD
  export_to_sheets(gc,ndf,worksheet_name=work_sheet,mode='a')

def save_todb(gc):
  ## create button that will append a row to the DB
  button = widgets.Button(description="Save me!")

  def on_button_clicked(b):
    sv.Inp_dict['UUID'] = sv.Inp_dict['Ticker'] + datetime.now().strftime('%Y%m%d%H%M%S')
    sv.Inp_dict['LastUpdate'] = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    append_todb(gc,dfts,'Ticker')
    append_todb(gc,dfls,'Lease')
    append_todb(gc,dfos,'Optionholdings')

  button.on_click(on_button_clicked)
  display(button)
  return button


def run_cashflow():
  ## create button
  button = widgets.Button(description="Cashflow Projections")
  out = widgets.Output()
  #display(out)
  def on_button_clicked(b):
    cfdict = dacf.calc_cashflow(sv.comp,sv.Inp_dict,sim={'Do':0,'Vol':5})
    #print(cfdict['cashflow'].transpose().round(2))
    with out:
      out.clear_output(True)
      display(pd.DataFrame(cfdict['Intrinsic_Price'],index=[sv.comp.ticksym + ' Intrinsic Share Price ']).style.format('{:.1f}'))
      display(widgets.HTML('<h4> Projeced Cashflows ($M) </h4>'))
      display(cfdict['tmp_cf'].style.format(cfdict['form_dict']))
      display('Current Actual Sale2cap','{:.2f}'.format(cfdict['curr_sales2cap']))
      dacf.sanity_checks(cfdict)
      dacf.mk_waterfall(cfdict['wf_dict'],divby=sv.comp.info['sharesOutstanding'])
    sv.Inp_dict['cfdict'] = cfdict
  button.on_click(on_button_clicked)
  display(button)
  return button, out

def run_cashflow_sim():
  ## create button
  button = widgets.Button(description="Simulate CF")
  out = widgets.Output()
  def on_button_clicked(b):
    with out:
      out.clear_output(True)
      sim_df = dacf.run_sim(sv.comp,sv.Inp_dict,nsim=100)
    sv.Inp_dict['sim_df'] = sim_df
  button.on_click(on_button_clicked)
  display(button)
  return button, out

def save_results(gc):
  ## create button
  button = widgets.Button(description="Save results")
  columns1 = ['UUID','Ticker','LastUpdate']
  columns2 = ['Price','Sim25','Sim75','CurrentPrice']
  res_dict = {}
  res_df = pd.DataFrame(columns=columns1+columns2)
  def on_button_clicked(b):
    for i in columns1: res_dict[i] = sv.Inp_dict[i]
    res_dict['FairPrice'] = sv.Inp_dict['cfdict']['equity_val_pershare']
    res_dict['Sim25'] = np.percentile(sv.Inp_dict['sim_df']['equity_val_pershare'],25)
    res_dict['Sim75'] = np.percentile(sv.Inp_dict['sim_df']['equity_val_pershare'],75)
    res_dict['CurrentPrice'] = sv.comp.info['previousClose']
    for i, (k,v) in enumerate(sv.Inp_dict['cfdict']['Intrinsic_Price'].items()): res_dict[k] = v
    res_df = pd.DataFrame(data=res_dict,index=[0])
    export_to_sheets(gc,res_df,worksheet_name='Results',mode='a')
  button.on_click(on_button_clicked)
  display(button)
  return button

def display_wids(DBdict):
  #tick_dict = get_ticker(DBdict)
  #lease_ui_dict, options_ui_dict = get_lease_opt()
  value_dict = value_inputs()
  sav2db = save_todb(gc)
  run_cf,cf_out = run_cashflow()
  run_cfsim, sim_out = run_cashflow_sim()
  sav_res = save_results(gc)
  display(cf_out)
  display(sim_out)
  #l = widgets.link((tick_dict['ui'], 'value'), (value_dict['ui'], 'value'))
  
  #display(tick_dict['ui'],tick_dict['out'])
  #display(lease_ui_dict['title'],lease_ui_dict['ui'],lease_ui_dict['out'])
  #display(options_ui_dict['title'],options_ui_dict['ui'],options_ui_dict['out'])
  #display(value_dict['title'],value_dict['ui'],value_dict['out'])
  #display(sav2db)
  return 
