import sys
import importlib
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import lognorm
import random
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import date
import pdb
import comp_data
#from yfinance 
import yfinance as yf

from fin_gists import shared_vars as sv

'''----// Calculate average revenue CAGR & EBIT margin //----'''
def get_cagr(comp):
  #CAGR = (past_revs.iloc[0]/past_revs.iloc[3])**(1/4)-1
  time_to_10K = (comp.quarterly_financials.columns[0]-comp.financials.columns[0]).days/365
  if time_to_10K >0.5: 
    revs_10K = comp.financials.loc['Total Revenue'].iloc[0]
  else: ## less than 1/2 year then the cagr may not be that representative so go to older 10K
    revs_10K = comp.financials.loc['Total Revenue'].iloc[1]
    time_to_10K += 1 # add a year
  revs_ttm = sum(comp.quarterly_financials.loc['Total Revenue'])
  cagr = (revs_ttm/revs_10K)**(1/time_to_10K)-1
  return(cagr)

def get_average_margin(past_ebit):
    margin = 0
    margin_lst = []
    for i in range(len(past_ebit.columns)):
        margin = past_ebit.iloc[1,i]/past_ebit.iloc[0,i]
    margin_lst.append(margin)
    return(sum(margin_lst)/len(margin_lst))

def get_market_info(ID,metric='long_tax_rate'):
  marketdata = comp_data.Market()
  met_df = pd.DataFrame()
  if metric=='long_tax_rate':
    country_df = marketdata.get_country_tax_rates()
  elif metric=='risk_premium_US':
    country_df = marketdata.get_risk_premiums_US()
  elif metric=='risk_premium':
    country_df = marketdata.get_risk_premiums()
  
  metdat_av=0
  #prev_year = str(int(date.today().strftime('%Y'))-1)
  prev_year = country_df.columns.max() #get the most recent year of data.
  for icont in ['Country1','Country2','Country3']:
    if metric=='long_tax_rate':
      tmpstr = country_df.loc[ID[icont]][prev_year].strip('%')
    elif metric=='risk_premium_US':
      tmpstr = country_df['implied premium (fcfe)'].loc[prev_year].strip('%')
    elif metric=='risk_premium':
      tmpstr = country_df.loc[country_df.index.str.replace(" ",'')
                              ==ID[icont].replace(" ",'')]['equity riskpremium'][0].strip('%')
    metdat = float(tmpstr)/100 if tmpstr else 0
    wts = ID[icont + 'Wt']
    metdat_av += metdat*wts
    met_df.loc[metric,ID[icont]] = metdat if tmpstr else 0
  return metdat_av, met_df

def get_industry_info(ID,metric='long_term_coc'):
  metdat_av=0
  met_df = pd.DataFrame()
  for iindt in ['Industry1','Industry2','Industry3']:
    inddata = comp_data.Industry(ID[iindt])
    percfac = 100
    if metric == 'long_term_coc':
      tmpstr = inddata.get_cost_of_capital().loc['cost of capital'].strip('%')
    elif metric == 'stddev':
      tmpstr = inddata.get_betas().loc['standard deviation of equity'].strip('%')
    elif metric == 'beta':
      tmpstr = inddata.get_betas().filter(regex='(^(beta))',axis=0)[0]
      percfac = 1
    elif metric=='debt_cost':
      tmpstr = inddata.get_cost_of_capital().loc['cost of debt'].strip('%')
    metdat = float(tmpstr)/percfac if tmpstr else 0
    wts = ID[iindt + 'Wt']
    metdat_av += metdat*wts
    met_df.loc[metric,ID[iindt]] = metdat if tmpstr else 0
  return metdat_av, met_df

'''----// Get WACC and net debt //----'''
def get_wacc(company_ticker="MSFT", market_risk_premium=0.059, debt_cost=0.02, tax_rate=0.3):
    risk_free = yf.Ticker('^TNX')
    risk_free_rate = risk_free.info['previousClose']/100

    if isinstance(company_ticker,str):
      comp = yf.Ticker(company_ticker)
    elif isinstance(company_ticker,yf.Ticker):
      comp = company_ticker
    
    prev_year = str(int(date.today().strftime('%Y'))-1)
    if 'Country1' in sv.Inp_dict:
      market_risk_premium, mkt_df = get_market_info(sv.Inp_dict,metric='risk_premium')
    else:
      market_risk_premium = float(comp.marketdata.get_risk_premiums_US()['implied premium (fcfe)'].loc[prev_year].strip('%'))/100
    
    if 'Industry1' in sv.Inp_dict:
      ## debt_return cost of debt industry specific. Interest expense
      debt_cost, debt_cf = get_industry_info(sv.Inp_dict,metric='debt_cost')
    
    equity_beta = comp.info['beta'] if comp.info['beta'] else comp.ind_beta ## does this need to be levered?
    equity_return = risk_free_rate+equity_beta*market_risk_premium
    market_cap = comp.info['marketCap']

    company_value = market_cap + comp.net_debt
    WACC = market_cap/company_value * equity_return + comp.net_debt/company_value * debt_cost * (1-tax_rate)
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
  rnd = comp.rnd
  tax_rate = comp.tax_rate
  nyrs = len(rnd)
  amort = rnd[1:]*(1/max(nyrs-1,1)) # current year from prev
  unamort = rnd*(np.arange(nyrs,0,-1)-1)/(nyrs-1)

  adj_op_income = comp.rnd[0]-sum(amort)
  rnd_asset = sum(unamort)

  #tax_rate = np.mean(comp.financials.loc['Income Tax Expense']/comp.financials.loc['Ebit'])
  tax_eff_rnd = adj_op_income*tax_rate
  rnd_dict = {'rnd_asset': rnd_asset, 'adj_op_income': adj_op_income, 'tax_eff_rnd': tax_eff_rnd}
  return rnd_dict

## options_conv
def option_conv(comp):
  opt_dict = sv.comp.opt_dict
  inddata = comp_data.Industry(sv.Inp_dict['Industry1']) 
  
  stddev, std_df = get_industry_info(sv.Inp_dict,metric='stddev')
  #stddev = float(inddata.get_betas().loc['standard deviation of equity'].strip('%'))
  # another source inddata.get_standard_deviation().loc['std deviation in equity']
  variance = stddev**2
  price = (comp.info['bid']+comp.info['ask'])/2
  strike = opt_dict['strike']
  divyield = comp.info['dividendYield'] if comp.info['dividendYield'] is not None else 0
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

## DDM Model
def ddm():
  curr_div = sv.comp.info['dividendYield']*sv.comp.info['previousClose']
  div_growth = rate_of_change(sv.Inp_dict['beg_div_growth'],sv.Inp_dict['year_conv'],sv.Inp_dict['long_div_growth'],sv.Inp_dict['terminal_year'],1)
  div_growth_cumm = (1+div_growth).cumprod()
  future_div = curr_div*div_growth_cumm
  
  cost_capital = rate_of_change(sv.Inp_dict['wacc'],sv.Inp_dict['year_conv'],sv.Inp_dict['long_term_coc'],sv.Inp_dict['terminal_year'],1)
  cost_capital_cumm = (1+cost_capital).cumprod()
  discount_factor = 1/cost_capital_cumm
  
  future_div_disc = future_div*discount_factor
  disc_divNyr = sum(future_div_disc)
  terminal_div = future_div_disc[-1] * (1 + sv.Inp_dict['long_div_growth'])
  terminal_div_disc = terminal_div/(sv.Inp_dict['wacc']-sv.Inp_dict['long_div_growth'])
  
  valuepershare = disc_divNyr + terminal_div_disc
  
  ddm_dict = locals()
  return ddm_dict

## NAV - cap rate model
def caprate_mod():
  NOI = sv.comp.gross_profit_margin # basically the gross profitcomp.ttm_revs
  NNOI = (NOI - sv.Inp_dict['maintcapex'])*sv.comp.ttm_revs
  ValRealEstateOps =  NNOI/sv.Inp_dict['cap_rate']
  ValueofEquity = ValRealEstateOps - (sv.comp.short_longterm_debt + sv.comp.longterm_debt) + sv.comp.cash_mms
  ValuePerShare = ValueofEquity/sv.comp.info['sharesOutstanding']
  
  caprate_nav_dict = locals()
  return caprate_nav_dict
  

## REIT adjustments in CF - add the few extra elements for reit
def cf_reit_adj(cashflow):
  da_rate = rate_of_change(sv.Inp_dict['darate'],sv.Inp_dict['year_conv'],sv.Inp_dict['long_da_rate'],sv.Inp_dict['terminal_year'],1)
  cashflow['EBITDA'] = cashflow['EBIT'] # for REITs the margin provided should be EBITDA margin
  cashflow['da'] = da_rate*cashflow['rev_fcst']
  cashflow['sbc'] = cashflow['rev_fcst']*sv.Inp_dict['sbc']
  cashflow['maintcapex'] = cashflow['rev_fcst']*sv.Inp_dict['maintcapex']
  interest_rate = rate_of_change(sv.Inp_dict['beg_int'],sv.Inp_dict['year_conv'],sv.Inp_dict['long_term_int'],sv.Inp_dict['terminal_year'],1)
  cashflow['interestexp'] = cashflow['rev_fcst']*interest_rate
  cashflow['EBT'] = cashflow['EBITDA'] -cashflow['da']- cashflow['interestexp'] # In a REIT Interest expense is part of the business model
  cashflow['EBTafterTax'] = cashflow['EBT']-(cashflow['EBT']-cashflow['NOL']).clip(lower=0)*cashflow['tax_rate']
  cashflow['EBITafterTax'] = cashflow['EBTafterTax'] ## for REIT we actually need the EBT - but doing this for downstreem calcs - fix later
  cashflow['FFO'] = cashflow['EBTafterTax'] - cashflow['Reinvestment'] + cashflow['da']
  cashflow['FCFF'] = cashflow['FFO'] + cashflow['sbc'] - cashflow['maintcapex'] ## this is equal to AFFO
  cashflow['shares'] = sv.comp.info['sharesOutstanding']*(1+sv.Inp_dict['stock_dilution_rate'])**np.arange(1,sv.Inp_dict['terminal_year']+1) # num shares will grow for new capital projects
  cashflow['PVFCFF'] = cashflow['FCFF']*cashflow['discount_factor']
  cashflow['PVFCFFpershare'] = cashflow['PVFCFF']/cashflow['shares'] # this can be used for IRR? This is actually FCFE - sinze it removes interest?
  return cashflow

def calc_cashflow(comp,ID,sim={'Do':0, 'Vol':5}):
  #locals().update(Inp_dict)
  
  rnd_dict = ID['rnd_dict']
  lease_dict = ID['lease_dict']
  ttm_revs = comp.ttm_revs
  tax_rate = ID['tax_rate']
  long_term_cagr = ID['long_term_cagr']
  long_term_margin = ID['long_term_margin']
  
  #inddata = comp.inddata
  marketdata = comp.marketdata
  
  #country_df = marketdata.get_country_tax_rates()
  #prev_year = str(int(date.today().strftime('%Y'))-1)
  #long_tax_rate = get_market_info(ID,metric='long_tax_rate')
  #float(country_df.loc[country_df.index.str.contains(comp.Country)][prev_year].values[0].strip('%'))/100 # for long term

  #long_tax_rate = comp.long_tax_rate  #get_market_info(ID,metric='long_tax_rate')
  
  #long_term_coc = float(inddata.get_cost_of_capital().loc['cost of capital'].strip('%'))/100 # sector specifc?
  long_term_coc = ID['long_term_coc'] #comp.long_term_coc  #get_industry_info(ID,metric='long_term_coc')
  wacc = ID['wacc'] #comp.wacc  #get_wacc(comp)

  if sim['Do']:
    long_term_margin = min(long_term_margin*1.5,create_rand(long_term_margin*sim['Vol']/15,long_term_margin)) #margin can't be >60%!!
    long_term_coc = create_rand(long_term_coc*sim['Vol'],long_term_coc)
    long_term_cagr = min(long_term_coc/2,create_rand(long_term_cagr*sim['Vol'],long_term_cagr)) #long term growth rate has to be lower than cost of capital or else infite growth
    
  rev_rate = rate_of_change(ID['beg_cagr'],ID['year_conv'],long_term_cagr,ID['terminal_year'],1)
  rev_cumrate = (1+rev_rate).cumprod()
  rev_fcst = ttm_revs*rev_cumrate
  
  tax_rate = rate_of_change(ID['tax_rate'],ID['year_conv'],ID['long_tax_rate'],ID['terminal_year'],1)
  
  margin_rate = rate_of_change(ID['beg_margin'],ID['year_conv'],long_term_margin,ID['terminal_year'],2)
  cost_capital = rate_of_change(wacc,ID['year_conv'],long_term_coc,ID['terminal_year'],1)
  cost_capital_cumm = (1+cost_capital).cumprod()
  discount_factor = 1/cost_capital_cumm

  debt_book_value = comp.short_longterm_debt + comp.longterm_debt
  equity_book_value = comp.quarterly_balance_sheet.loc['Total Stockholder Equity'].iloc[0]
  invested_capital = (equity_book_value + debt_book_value 
                      + rnd_dict['rnd_asset'] 
                      - lease_dict['debt_value_lease']
                      - comp.cash_mms)
  curr_sales2cap = ttm_revs/invested_capital
  
  '''--------// Build the Cashflow //-----'''
  cashflow = pd.DataFrame()
  cashflow['rev_rate'] = rev_rate
  cashflow['rev_fcst'] = rev_fcst
  cashflow['margin_rate'] = margin_rate
  cashflow['EBIT'] = rev_fcst*margin_rate
  cashflow['tax_rate'] = tax_rate
  cashflow['Reinvestment'] = np.diff(np.append([ttm_revs],rev_fcst))/ID['sales_to_capital'] if ID['sales_to_capital'] else 0
  NOL = [float(ID['NOLbase'])*1e6]
  InvCap = [invested_capital]
  for inol in range(ID['terminal_year']):
    NOL.append(max(NOL[inol]-cashflow['EBIT'].iloc[inol],0))
    InvCap.append(InvCap[inol]+cashflow['Reinvestment'].iloc[inol])
  NOL.pop(0)
  InvCap.pop(0)
  cashflow['NOL'] = NOL
  cashflow['InvestedCapital'] = InvCap

  cashflow['cost_capital'] = cost_capital
  cashflow['discount_factor'] = discount_factor
  
  if sv.Inp_dict['Industry1'] == 'R.E.I.T.':
    cashflow = cf_reit_adj(cashflow)
  else:
    cashflow['EBITafterTax'] = cashflow['EBIT']-(cashflow['EBIT']-cashflow['NOL']).clip(lower=0)*cashflow['tax_rate']
    cashflow['FCFF'] = cashflow['EBITafterTax'] - cashflow['Reinvestment']
    cashflow['PVFCFF'] = cashflow['FCFF']*cashflow['discount_factor']

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
  value_of_OpAss = pv_totalCF*(1-ID['prob_failure']) + liquid_val*ID['prob_failure']*ID['distress_price']
  equity_value = (value_of_OpAss - debt_book_value 
                  - lease_dict['debt_value_lease']  
                  - ID['minority_interest']*1e6 
                  + comp.cash_mms
                  + ID['crossholdings_nonopassets']*1e6)
  value_equity_commonstock = equity_value - ID['value_op_outstanding']
  equity_val_pershare = value_equity_commonstock/comp.info['sharesOutstanding']
  if not sim['Do']: # if not simulation do some plotting and printing
    Intrinsic_Price = {'DCF': equity_val_pershare}
    listofmill = []
    listofnum = []
    ## if REIT run the ddm and nav model
    if sv.Inp_dict['Industry1'] == 'R.E.I.T.':
      caprate_nav_dict = caprate_mod()
      ddm_dict = ddm()  
      Intrinsic_Price['DDM'] = ddm_dict['valuepershare']
      Intrinsic_Price['navCapRate'] = caprate_nav_dict['ValuePerShare']
      listofmill = ['EBITDA',	'da',	'sbc',	'maintcapex',	'interestexp',	'EBT',	'EBTafterTax',
                    'EBITafterTax',	'FFO']
      listofnum  = ['shares','PVFCFFpershare']
    #sanity checks
    #print pretty cashflow - $s in mill others in %
    tmp_cf = cashflow.copy()
    listofmill = listofmill + ['rev_fcst','EBIT','Reinvestment','NOL','EBITafterTax','FCFF','PVFCFF','InvestedCapital']
    form_dict = dict(zip(listofmill+listofnum,["{:,.0f}"]*len(listofmill+listofnum)))
    perclist = list(set(cashflow.columns.tolist())-set(listofmill+listofnum))
    percdict = dict(zip(perclist,["{:.1f}%"]*len(perclist)))
    tmp_cf[listofmill] = cashflow[listofmill]/1e6
    tmp_cf[perclist] = cashflow[perclist]*100
    #print(tmp_cf)
    #format_mapping = {"Currency": "${:,.2f}", "Int": "{:,.0f}", "Rate": "{:.2f}%"}
    form_dict.update(percdict)
    #display(tmp_cf.style.format(form_dict))
    #print waterfall
    wf_dict = {'pv_terminal_value': [pv_terminal_value, 'relative'],
               'pv_CFNyr': [pv_CFNyr, 'relative'],
               'pv_totalCF': [pv_totalCF, 'total'],
               'liquidation effect': [-pv_totalCF+value_of_OpAss,'relative'],
               'value_of_OpAss': [value_of_OpAss, 'total'],
               'debt_book_value': [-debt_book_value, 'relative'],
               'debt_value_lease': [-lease_dict['debt_value_lease'],'relative'],
               'minority_interest': [-ID['minority_interest']*1e6,'relative'],
               'cash&mms': [comp.cash_mms,'relative'],
               'crossholdings_nonopassets': [ID['crossholdings_nonopassets']*1e6,'relative'],
               'equity_value': [equity_value, 'total'],
               'value_options_outstanding': [-ID['value_op_outstanding'],'relative'],
               'value_equity_commonstock': [value_equity_commonstock, 'total']
              }
  cfdict = locals()
  return cfdict

def mk_waterfall(wf_dict,divby=1e6):
  import plotly.graph_objects as go
  wf_dict = {k:[v[0]/divby,v[1]] for k,v in wf_dict.items()} 
  #print(wf_dict)
  fig = go.Figure(go.Waterfall(
      name = "WF", orientation = "v",
      measure = [v[1] for k,v in wf_dict.items()],
      x = [k for k,v in wf_dict.items()], 
      textposition = "outside",
      text = ["{:0.1f}".format(v[0]) for k,v in wf_dict.items()],
      y = [v[0] for k,v in wf_dict.items()],
      connector = {"line":{"color":"rgb(63, 63, 63)"}},
  ))

  fig.update_layout(
          title = "Equity Value Water Fall",
          showlegend = True,
          width = 800
  )
  fig.show()
  return
  
def plot_sim(sim_df,cfdict,comp=sv.comp):
  ## Plot the histogram
  fig, ax = plt.subplots(1, 1)
  sns.set_style('darkgrid')
  sns.histplot(sim_df['equity_val_pershare'],stat='probability')
  ax.text(comp.info['previousClose'],0.2,'* prevClose',rotation=60)
  ax.text(cfdict['equity_val_pershare'],0.2,'* Intrinsic Value',rotation=60)
  ax.plot([comp.info['fiftyTwoWeekHigh'],comp.info['fiftyTwoWeekLow']],[.10,.10],color='r')
  plt.show();
  return ax

def run_sim(comp,Inp_dict,nsim=100):
  simlist = []
  selcols = ['equity_val_pershare', 'value_equity_commonstock','long_term_coc','long_term_margin',
             'long_term_cagr','terminal_value',  'pv_totalCF',
             'pv_CFNyr', 'pv_terminal_value']
  for irnd in range(nsim):
    simdict = calc_cashflow(comp,Inp_dict,sim={'Do':1,'Vol':5})
    rowlist=[]
    for isel in selcols:
      rowlist.append(simdict[isel])
    simlist.append(rowlist)
  sim_df = pd.DataFrame(simlist,columns=selcols)
  cfdict = calc_cashflow(comp,Inp_dict,sim={'Do':0,'Vol':5})
  plot_sim(sim_df,cfdict,sv.comp)
  sim_df[['long_term_cagr','long_term_coc','long_term_margin']].plot(title='Simulation variables')
  plt.show()
  return sim_df

def sanity_checks(cfdict):
  #compare the current and projected numbers against peer group
  ## 10 yr
  listind = ['Industry1','Industry2','Industry3']
  indlist_tmp = [sv.Inp_dict[i] for i in listind]
  indlist = list(filter(None, indlist_tmp))
  paramlist = ['revenue','equity','ROE','ROIC']
  paramformat = ["${:,.0f}"]*2 + ["{:,.1%}"]*2
  ## check if cfdict has sandf - if so check if the industries have been done
  ## if so don't pull the same industries again. - code all this later
  sandf = pd.DataFrame(index = paramlist,columns=['Current','10th Year']+indlist)
  sandf.loc[:,'10th Year'] = [cfdict['cashflow']['rev_fcst'].iloc[-1]/1e6, 
                     cfdict['value_equity_commonstock']/1e6,
                     cfdict['cashflow']['EBITafterTax'].iloc[-1]/cfdict['value_equity_commonstock'],
                     cfdict['cashflow']['ROIC'].iloc[-1]]
  sandf.loc[:,'Current'] = [cfdict['cashflow']['rev_fcst'].iloc[0]/1e6, 
                   cfdict['equity_book_value']/1e6,
                   cfdict['cashflow']['EBITafterTax'].iloc[0]/cfdict['equity_book_value'], 
                   cfdict['cashflow']['ROIC'].iloc[0]]
  ##get industry data
  
  for iindt in indlist:
    inddata = comp_data.Industry(iindt)
    cash_arr = inddata.get_cash().squeeze()
    ind_cash = float(cash_arr['cash (us $ millions)'].replace('$','').replace(',',''))
    tmpcoc = float(inddata.get_cost_of_capital().loc['cost of capital'].strip('%'))/100
    tmproe = float(inddata.get_roe().loc['roe (adjusted for r&d)'].strip('%'))/100
    EBIT = float(inddata.get_margins().loc['pre-tax lease & r&d adj margin'].strip('%'))
    sandf.loc[:,iindt] = [ind_cash*100./float(cash_arr['cash/revenues'].strip('%')),
                       ind_cash*100./float(cash_arr['cash/firm value'].strip('%')),
                       tmproe, tmpcoc] 
  #df['Industry US] = {'revenue10thyr': cfdict['cashflow'].loc['rev_fcst'][-1], 
  display(sandf.transpose().style.format(dict(zip(paramlist,paramformat))))
  cfdict['sandf'] = sandf
  return
  #revenue
  
def damoCF():
  return

if __name__ == "__main__":
  damoCF()
