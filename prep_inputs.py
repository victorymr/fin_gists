import sys
import importlib
import pandas as pd
import numpy as np

from datetime import date
import pdb
import comp_data
from yfinance import yfinance as yf

import ipywidgets as widgets
from IPython.display import display

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials
import gspread_dataframe
from gspread_dataframe import get_as_dataframe, set_with_dataframe

from fin_gists import damoCF as dacf

style = {'description_width': 'initial'}

def read_sheet(workbook,sheet):
  ws = workbook.worksheet(sheet)
  df = get_as_dataframe(ws)
  df.replace(np.nan, '', regex=True,inplace=True)
  
def read_DB(gc,filn='StockDB'):
  ## This function can be made more elegant and streamlined to automatically identify the sheets and download in dfs/dict
  workbook = gc.open('StockDB')
  sheets = workbook.worksheets()
  #sheets = ["Ticker",'Lease','Optionholdings']
  DBdict = {isheet: read_sheet(workbook,isheet) for isheet in sheets}
  return DBdict
  
def get_ticker(DBdict):
  ## Enter your Ticker Symbol - make sure no mistakes!!
  ## If this symbol exists in my DB - I am going to get its latest data else empty df
  ## populate the cells - either from spreadhseet or calculations or defaults
  ## Where the spreadsheet populates but we have more recent data - print the new data to allow user to decide
  
  ticksym = widgets.Text(
      value='MSFT',
      placeholder='Type something',
      description='Ticker Symbol:',
      disabled=False,
      style=style
  )
  def f(ticksym):
    ## If this symbol exists in my DB - I am going to get its latest data
    #global dfts, dfls, dfos

    dftickset = dft[dft['Ticker']==ticksym]
    if len(dftickset):
      dfts = dftickset.iloc[-1]
      dfls = dfl[dfl['UUID'].astype(str)==dfts['UUID']]
      dfos = dfo[dfo['UUID'].astype(str)==dfts['UUID']]
      print(ticksym, ' Last Updated on ', dfts['LastUpdate'])
    else:
      dfts = dft.loc[0].copy() #default take the first row
      dfls = dfl.loc[0].copy()
      dfos = dfo.loc[0].copy()
      print(ticksym, ' NOT FOUND in DB - Using defaults please update appropriately')
    # get comp info
    comp = yf.Ticker(ticksym.value)
    ttm_revs = sum(comp.quarterly_financials.loc['Total Revenue']) #
    ttm_ebit = sum(comp.quarterly_financials.loc['Ebit'])
    cash_mms = comp.quarterly_balance_sheet.loc['Cash'].iloc[0]+comp.quarterly_balance_sheet.loc['Short Term Investments'].iloc[0]
    net_debt = comp.quarterly_balance_sheet.loc['Short Long Term Debt'].iloc[0] + comp.quarterly_balance_sheet.loc['Long Term Debt'].iloc[0] - cash_mms
    interest_expense = sum(comp.quarterly_financials.loc['Interest Expense'])/net_debt
    tax_rate = np.mean(comp.financials.loc['Income Tax Expense']/comp.financials.loc['Ebit']) # avg over past few years
    rnd_dict = dacf.rnd_conv(comp)
    curr_cagr = dacf.get_cagr(comp)
    comp.marketdata = marketdata

  out = widgets.interactive_output(f, {'ticksym': ticksym})
  display(ticksym,out)
  
def get_lease_opt():
  ## get the lease input and options data
  ## Make a widgets and prepopulate them? Allow user to modify if need be
  ## 
  lslist = [widgets.FloatText(description = 'Year' + str(i), value = dfls['Year'+str(i)],readout_format=':,.2f') for i in range(6)]
  lstup = ['Year' + str(i) for i in range(6)]
  lsdict = dict(zip(lstup,lslist))
  lsdict['bulk_commitment'] = widgets.FloatText(description = 'bulk_commitment', value = dfls['YearBulk'],style=style)
  lsdict['nyrs_bulk'] = widgets.FloatText(description = 'N yrs bulk', value = dfls['nBulk'],style=style)
  lsdict['cost_of_debt'] = widgets.FloatText(description = 'debt cost', value = dfls['cost_of_debt'],style=style)

  #globals().update(lsdict)
  lease_inp['Year'] = np.arange(1,6)
  def flease(**lsdict):
    lease_inp['current_commitment'] = lsdict[lstup[0]]
    lease_inp['future_commitments'] = [lsdict[lstup[i]] for i in range(1,6)]
    lease_inp['bulk_commitment'] = lsdict['bulk_commitment']
    lease_inp['nyrs_bulk'] = lsdict['nyrs_bulk']
    lease_inp['cost_of_debt'] = lsdict['cost_of_debt']
    lease_dict = dacf.lease_conv(lease_inp)
    ebit_adj = ttm_ebit + lease_dict['adj_op_income'] + rnd_dict['adj_op_income']
    mean_margin = ebit_adj/ttm_revs

    #print(lstup)
  
  layout =widgets.Layout(grid_template_columns='1fr 1fr 1fr')
  lsui = widgets.GridBox( tuple(lsdict.values()),layout = layout)
  lsout = widgets.interactive_output(flease, lsdict)
  ltit = widgets.HTML('<h4> Lease Commitment Inputs ($M) </h4>')
  display(ltit)
  display(lsui, lsout)

  ## Options inputs
  opdict = {}
  opdict['strike'] = widgets.FloatText(description = 'Avg Strike', value = dfos['Strike'][0],style=style)
  opdict['expiration'] = widgets.FloatText(description = 'Avg Expiration', value = dfos['AvgMaturity'][0],style=style)
  opdict['n_options'] = widgets.FloatText(description = 'Num of Options', value = dfos['NumOptions'][0],style=style)
  def foptions(**opdict):
    opt_dict['strike'] = opdict['strike']
    opt_dict['expiration'] = opdict['expiration']
    opt_dict['n_options'] = opdict['n_options']
    comp.opt_dict = opt_dict

  opui = widgets.GridBox(tuple(opdict.values()),layout = layout)
  opout = widgets.interactive_output(foptions, opdict)
  otit = widgets.HTML('<h4> Options Outstanding Inputs </h4>')
  display(otit)
  display(opui, opout)

def value_inputs():
  country_name_list = list(marketdata.get_country_tax_rates().index) + ['']
  industry_name_list = comp_data.industry_name_list + ['']
  dfts_list = dfts.index
  dfts_res = [i for i in dfts_list if 'Unnamed' not in i] 
  lsdts_int = ['year_conv',	'terminal_year']
  lsdts_flt1 = ['beg_cagr',	'long_term_cagr',	'beg_margin',	'long_term_margin',	'tax_rate']
  lsdts_flt2 = ['sales_to_capital', 'minority_interest',	'crossholdings_nonopassets' ]
  lsdts_indt =['Industry1',	'Industry2', 'Industry3']
  lsdts_indf = ['Industry1Wt',	'Industry2Wt', 'Industry3Wt']
  lsdts_cont = ['Country1','Country2','Country3']
  lsdts_conf = ['Country1Wt','Country2Wt','Country3Wt']
  lsdts_liqp = ['prob_failure','distress_price']
  lsdts_liqt = ['liquidation_type']

  dfts_dict = {i: widgets.IntText(description=i,value=dfts[i],style=style) for i in lsdts_int }
  dfts_dict.update({'Forecast': widgets.HTML('<b>Time Horizon</b>')})
  dfts_dict.update({'GrowthMargins': widgets.HTML('<b>Growth & Margins</b>')})
  dfts_dict.update({i: widgets.FloatSlider(description=i,min=0,max=1,value=dfts[i],style=style) for i in lsdts_flt1})
  dfts_dict.update({i: widgets.FloatText(description=i,value=dfts[i],style=style) for i in lsdts_flt2})
  dfts_dict.update({i: widgets.Dropdown(options=industry_name_list, description=i,value=dfts[i],style=style) for i in lsdts_indt})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1, description=i,value=dfts[i],style=style) for i in lsdts_indf})
  dfts_dict.update({i: widgets.Dropdown(options=country_name_list, description=i,value=dfts[i],style=style) for i in lsdts_cont})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1, description=i,value=dfts[i],style=style) for i in lsdts_conf})
  dfts_dict.update({i: widgets.FloatSlider(min=0,max=1, description=i,value=dfts[i],style=style) for i in lsdts_liqp})
  dfts_dict.update({i: widgets.Dropdown(options=[('Fair Value', 'V'), ('Book Value', 'B')], description=i,value=dfts[i],style=style) for i in lsdts_liqt})

  def finpdict(**dfts_dict):
    for k,v in dfts_dict.items():
      Inp_dict[k] = v

    indt_list = [v for k,v in Inp_dict.items() if k in lsdts_indt]
    ind_df = pd.DataFrame(columns=indt_list)
    for iindt in indt_list:
      ind_dat = comp_data.Industry(iindt)
      tmp_df = ind_dat.get_historical_growth()
      tmp_df = tmp_df.append(ind_dat.get_margins()[['net margin','pre-tax unadjusted operating margin','pre-tax lease & r&d adj margin']])
      tmp_df.loc['sales/capital'] = ind_dat.get_capital_expenditures()['sales/capital']
      tmp_df = tmp_df.append(ind_dat.get_industry_tax_rates()[['average across only money-making companies2','aggregate tax rate3']])
      ind_df[iindt] = tmp_df

    ## Relevant Metrics from Company's recent financials
    display(widgets.HTML('<h4> Metrics from Company Recent Financials </h4>'))
    listvar = ['ebit_adj','ttm_ebit','mean_margin','curr_cagr',
               'interest_expense','tax_rate']
    list_dict = {i:'{:,.2f}'.format(eval(i)) for i in listvar}
    display(pd.DataFrame(data=list_dict.values(),
                         index=list_dict.keys(),columns=[ticksym.value]))

    ## Relevant Industry Metrics
    display(widgets.HTML('<h4> Key Industry Metrics - Use as Reference </h4>'))
    display(ind_df)

    ## Relevant Country of operation Metrics
    prev_year = str(int(datetime.today().strftime('%Y'))-1)
    display(widgets.HTML(value='<h4> Key Country Level Metrics from ' + prev_year + ' - Use as Reference </h4>'))
    cont_list = [(v, marketdata.get_country_tax_rates().loc[v,prev_year]) for k,v in Inp_dict.items() if k in lsdts_cont]
    display(('Tax Rates ',cont_list))

  layout =widgets.Layout(grid_template_columns='1fr 1fr 1fr')
  dfts_ui = widgets.GridBox( tuple(dfts_dict.values()),layout = layout)
  dfts_out = widgets.interactive_output(finpdict, dfts_dict)
  inptit = widgets.HTML('<h2> Key Value Inputs </h2>')
  display(inptit)
  display(dfts_ui, dfts_out)
