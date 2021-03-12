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
def get_wacc(company_ticker="MSFT", market_risk_premium=0.059, debt_return=0.01, tax_rate=0.3):
    risk_free = yf.Ticker('^TNX')
    risk_free_rate = risk_free.info['previousClose']/100

    prev_year = str(int(date.today().strftime('%Y'))-1)
    market_risk_premium = float(comp_data.Market().get_risk_premiums_US()['implied premium (fcfe)'].loc[prev_year].strip('%'))/100

    if isinstance(company_ticker,str):
      comp = yf.Ticker(company_ticker)
    elif isinstance(company_ticker,yf.Ticker):
      comp = company_ticker

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

def damoCF():
  return

if __name__ = "__main__":
  damoCF()
