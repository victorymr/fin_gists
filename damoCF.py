import sys
import importlib
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import lognorm
import random

from datetime import date
import pdb
#from compdata import comp_data
from yfinance import yfinance as yf

'''----// Calculate average revenue CAGR & EBIT margin //----'''
def get_cagr(comp):
  #CAGR = (past_revs.iloc[0]/past_revs.iloc[3])**(1/4)-1
  time_to_10K = (comp.quarterly_financials.columns[0]-comp.financials.columns[0]).days/365
  revs_ttm = sum(comp.quarterly_financials.loc['Total Revenue'])
  revs_10K = comp.financials.loc['Total Revenue'].iloc[0]
  cagr = (revs_ttm/revs_10K)**(1/time_to_10K)-1
  return(cagr)

def get_average_margin(past_ebit):
    margin = 0
    margin_lst = []
    for i in range(len(past_ebit.columns)):
        margin = past_ebit.iloc[1,i]/past_ebit.iloc[0,i]
    margin_lst.append(margin)
    return(sum(margin_lst)/len(margin_lst))
  
'''----// Get WACC and net debt //----'''
def get_wacc(company_ticker="MSFT", market_risk_premium=0.059, debt_return=0.02, tax_rate=0.3):
    risk_free = yf.Ticker('^TNX')
    risk_free_rate = risk_free.info['previousClose']/100


    if isinstance(company_ticker,str):
      comp = yf.Ticker(company_ticker)
    elif isinstance(company_ticker,yf.Ticker):
      comp = company_ticker
    
    
    prev_year = str(int(date.today().strftime('%Y'))-1)
    market_risk_premium = float(comp.marketdata.get_risk_premiums_US()['implied premium (fcfe)'].loc[prev_year].strip('%'))/100

    equity_beta = comp.info['beta']
    equity_return = risk_free_rate+equity_beta*market_risk_premium

    net_debt = comp.balance_sheet.loc['Short Long Term Debt'] + comp.balance_sheet.loc['Long Term Debt'] - comp.balance_sheet.loc['Cash']
    market_cap = comp.info['marketCap']

    company_value = market_cap + net_debt.iloc[0]
    WACC = market_cap/company_value * equity_return + net_debt.iloc[0]/company_value * debt_return * (1-tax_rate)
    return WACC

## growth patterns
def rate_of_change(beg_rate,year_conv,long_term_rate,terminal_year=10,change_type=1):
  rate_arr = np.zeros([terminal_year])
  if change_type==1: #flat and taper to long term- typical for revenue
    rate_arr[0:year_conv]=beg_rate
    rate_arr[year_conv:terminal_year] = np.interp(range(year_conv,terminal_year),[year_conv,terminal_year],[beg_rate,long_term_rate])
  elif change_type==2: # grow and flatten
    rate_arr[0:year_conv]= np.interp(range(0,year_conv),[0,year_conv],[beg_rate,long_term_rate])
    rate_arr[year_conv:terminal_year] = long_term_rate

  return rate_arr

##Lease converter
## leases are like debt
def lease_conv(lease_inp,cost_of_debt=0.03,nyrs_bulk=3):
  ##lease_inp - df with current and next few years of lease amounts
  cost_of_debt = lease_inp['cost_of_debt']
  nyrs_bulk = lease_inp['nyrs_bulk']
  pv_commitment = lease_inp['future_commitments']/(1+cost_of_debt)**lease_inp['Year']
  nyrs = len(pv_commitment)
  bulk_comm = lease_inp['bulk_commitment']/nyrs_bulk
  pv_bulk = bulk_comm*(1-(1+cost_of_debt)**(-nyrs_bulk))/(cost_of_debt*(1+cost_of_debt)**nyrs)
  debt_value_lease = sum(pv_commitment) + pv_bulk
  dep_op_lease = debt_value_lease/(nyrs+nyrs_bulk)
  adj_op_income = lease_inp['current_commitment'] - dep_op_lease

  lease_dict = {'debt_value_lease':debt_value_lease,\
                'adj_op_income':adj_op_income,'dep_op_lease': dep_op_lease}
  return lease_dict

## R&D converter
#3 we want to convert it from a opex to capital asset
def rnd_conv(comp):
  rnd = comp.financials.loc['Research Development']
  nyrs = len(rnd)
  amort = rnd[1:]*(1/max(nyrs-1,1)) # current year from prev
  unamort = rnd*(np.arange(nyrs,0,-1)-1)/(nyrs-1)

  adj_op_income = rnd[0]-sum(amort)
  rnd_asset = sum(unamort)

  tax_rate = np.mean(comp.financials.loc['Income Tax Expense']/comp.financials.loc['Ebit'])
  tax_eff_rnd = adj_op_income*tax_rate
  rnd_dict = {'rnd_asset': rnd_asset, 'adj_op_income': adj_op_income, 'tax_eff_rnd': tax_eff_rnd}
  return rnd_dict

## options_conv
def option_conv(comp):
  opt_dict = comp.opt_dict
  inddata = comp.inddata
  stddev = float(inddata.get_betas().loc['standard deviation of equity'].strip('%'))
  # another source inddata.get_standard_deviation().loc['std deviation in equity']
  variance = stddev**2
  price = (comp.info['bid']+comp.info['ask'])/2
  strike = opt_dict['strike']
  divyield = comp.info['dividendYield']
  expiration = opt_dict['expiration']
  n_options = opt_dict['n_options']
  n_shares = comp.info['sharesOutstanding']

  risk_free = yf.Ticker('^TNX')
  risk_free_rate = risk_free.info['previousClose']/100
  
  adj_K = strike
  adj_yield = risk_free_rate - divyield

  opt_value  = max(0,price-strike)
  ##iteration through this to get the opt_value 
  for it in range(200):
    adj_S = (n_shares*price + n_options*opt_value)/(n_shares + n_options)
    
    d1 = (np.log(adj_S/adj_K)+(adj_yield+variance/2)*expiration)/(variance*expiration)**0.5
    d2 = d1 - (variance*expiration)**2
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    opt_value = np.exp(-divyield*expiration)*adj_S*Nd1 - adj_K*np.exp(-risk_free_rate*expiration)*Nd2

  value_op_outstanding = opt_value*n_options
  return value_op_outstanding

def create_rand(s,l,v=0,type='lognorm'):
  if type=='lognorm':
    rv = lognorm(s,v,l)
  outrand = rv.ppf(random.random()) 
  return outrand
 
    
def calc_cashflow(comp,ID,sim={'Do':0, 'Vol':5}):
  #locals().update(Inp_dict)
  
  rnd_dict = ID['rnd_dict']
  lease_dict = ID['lease_dict']
  ttm_revs = ID['ttm_revs']
  tax_rate = ID['tax_rate']
  long_term_cagr = ID['long_term_cagr']
  long_term_margin = ID['long_term_margin']
  
  inddata = comp.inddata
  marketdata = comp.marketdata
  country_df = marketdata.get_country_tax_rates()
  
  long_tax_rate = float(country_df.loc[country_df.index.str.contains(comp.Country)].strip('%'))/100 # for long term
  long_term_coc = float(inddata.get_cost_of_capital().loc['cost of capital'].strip('%'))/100 # sector specifc?
  
  #pdb.set_trace()
  wacc = get_wacc(comp)

  if sim['Do']:
    long_term_margin = min(0.6,create_rand(long_term_margin*sim['Vol']/5,long_term_margin)) #margin can't be >60%!!
    long_term_coc = create_rand(long_term_coc*sim['Vol'],long_term_coc)
    long_term_cagr = min(long_term_coc/2,create_rand(long_term_cagr*sim['Vol'],long_term_cagr)) #long term growth rate has to be lower than cost of capital or else infite growth
    
  rev_rate = rate_of_change(ID['beg_cagr'],ID['year_conv'],long_term_cagr,ID['terminal_year'],1)
  rev_cumrate = (1+rev_rate).cumprod()
  rev_fcst = ttm_revs*rev_cumrate
  
  tax_rate = rate_of_change(ID['tax_rate'],ID['year_conv'],long_tax_rate,1)
  
  margin_rate = rate_of_change(ID['beg_margin'],ID['year_conv'],long_term_margin,ID['terminal_year'],2)
  cost_capital = rate_of_change(wacc,ID['year_conv'],long_term_coc,ID['terminal_year'],1)
  cost_capital_cumm = (1+cost_capital).cumprod()
  discount_factor = 1/cost_capital_cumm

  debt_book_value = (comp.quarterly_balance_sheet.loc['Long Term Debt'].iloc[0] 
                     + comp.quarterly_balance_sheet.loc['Short Long Term Debt'].iloc[0]
                     )
  equity_book_value = comp.quarterly_balance_sheet.loc['Total Stockholder Equity'].iloc[0]
  cash_mms = comp.quarterly_balance_sheet.loc['Cash'].iloc[0]+comp.quarterly_balance_sheet.loc['Short Term Investments'].iloc[0]
  invested_capital = (equity_book_value + debt_book_value 
                      + rnd_dict['rnd_asset'] 
                      - lease_dict['debt_value_lease']
                      - cash_mms)
  curr_sales2cap = ttm_revs/invested_capital
  
  '''--------// Build the Cashflow //-----'''
  cashflow = pd.DataFrame()
  cashflow['rev_rate'] = rev_rate
  cashflow['rev_fcst'] = rev_fcst
  cashflow['margin_rate'] = margin_rate
  cashflow['EBIT'] = rev_fcst*margin_rate
  cashflow['tax_rate'] = tax_rate
  cashflow['Reinvestment'] = np.diff(np.append([ttm_revs],rev_fcst))/ID['sales_to_capital']
  NOL = [ID['NOLbase']]
  InvCap = [invested_capital]
  for inol in range(ID['terminal_year']):
    NOL.append(max(NOL[inol]-cashflow['EBIT'].iloc[inol],0))
    InvCap.append(InvCap[inol]+cashflow['Reinvestment'].iloc[inol])
  NOL.pop(0)
  InvCap.pop(0)
  cashflow['NOL'] = NOL
  cashflow['EBITafterTax'] = cashflow['EBIT']-(cashflow['EBIT']-cashflow['NOL']).clip(lower=0)*cashflow['tax_rate']
  cashflow['FCFF'] = cashflow['EBITafterTax'] - cashflow['Reinvestment']
  cashflow['cost_capital'] = cost_capital
  cashflow['discount_factor'] = discount_factor
  cashflow['PVFCFF'] = cashflow['FCFF']*cashflow['discount_factor']

  cashflow['InvestedCapital'] = InvCap
  cashflow['ROIC'] = cashflow['EBITafterTax']/cashflow['InvestedCapital']


  # Calculate terminal value
  terminal_cashflow = cashflow['FCFF'].iloc[-1] * (1 + long_term_cagr)
  terminal_value = terminal_cashflow / (long_term_coc-long_term_cagr)
  # PV of Terminal Value
  pv_terminal_value = terminal_value * cashflow['discount_factor'].iloc[-1]
  pv_CFNyr = sum(cashflow['PVFCFF'])
  pv_totalCF = pv_CFNyr + pv_terminal_value
  if ID['liquidation_type']=="V":
    liquid_val = pv_totalCF
  else:
    liquid_val =  equity_book_value + debt_book_value
  value_of_OpAss = pv_totalCF + liquid_val*ID['prob_failure']*ID['distress_price']
  equity_value = (value_of_OpAss - debt_book_value 
                  - lease_dict['debt_value_lease']  
                  - ID['minority_interest'] 
                  + cash_mms
                  + ID['crossholdings_nonopassets'])
  value_equity_commonstock = equity_value - ID['value_op_outstanding']
  equity_val_pershare = value_equity_commonstock/comp.info['sharesOutstanding']
  
  cfdict = locals()
  return cfdict

def damoCF():
  return

if __name__ == "__main__":
  damoCF()
