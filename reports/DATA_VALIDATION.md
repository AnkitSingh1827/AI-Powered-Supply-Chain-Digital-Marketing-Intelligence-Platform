# Data Validation Notes

## Supplied files

| File | Rows | Key fields | Role |
|---|---:|---|---|
| supply_chain.csv | 5,000 | Date, ports, mode, risk, lead time, disruption target | Main disruption model |
| news_sentiment.csv | 4,845 | sentiment, text | NLP sentiment/event signal |
| marketing_campaign.csv | 200,000 | date, channel, ROI, clicks, conversions | Marketing/ROI layer |
| shipping_rates.csv | 300 | monthly date, freight/BDI/pressure | Historical trade features |
| port_congestion.csv | 6,260 | weekly port congestion | Historical logistics context |
| tariffs | 71 | date, tariff rate, policy | Policy risk context |
| trade_flows.csv | 1,250 | year, exporter/importer, trade value | Historical trade context |
| disruption_events.csv | 58 | date, event, severity | Historical event reference |
| commodity_prices_supply_chain.csv | 110,515 | date, commodity, price | Historical commodity context |
| industry_exposure.csv | 250 | year, industry, vulnerability | Industry risk context |

## Critical integration finding

`news_sentiment.csv` has no date column. Do not create a fake date. A valid
early-warning experiment needs dated news. The current project therefore keeps
NLP processing separate until dated news is available.

`marketing_campaign.csv` covers 2021, while `supply_chain.csv` covers 2024-2025.
Without a common product/customer/business key, the marketing data should not be
treated as a row-level training feature for the shipment target.
