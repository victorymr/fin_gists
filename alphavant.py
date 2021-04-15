AV_rapid_api = "a507037947msh01856e457ba4d92p151c90jsn493503c182d2"

!git clone https://github.com/RomelTorres/alpha_vantage.git
import alpha_vantage as av

from alpha_vantage.alpha_vantage.timeseries import TimeSeries
from alpha_vantage.alpha_vantage.fundamentaldata import FundamentalData as FD

fd = FD(key=AV_rapid_api, output_format='pandas',rapidapi=True)

inc_stat = fd.get_income_statement_annual('AMT')
bal_sheet = fd.get_balance_sheet_quarterly('AMT')
cash_flow = fd.get_cash_flow_annual('AMT')
