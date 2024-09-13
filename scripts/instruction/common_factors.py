"""
Common factor fine-tuning dataset.
"""

common_factor_fine_tuning_dataset = {
    "size": {
        "strategy_name": "size",
        "rationale": "This strategy is based on the size factors. \
Size factors are negatively correlated with future cryptocurrency cumulative returns. \
The size factor reflects the size effect of assets, which are in \
line with two mechanisms. First, size may proxy for an illiquidity premium. There are \
three sets of evidence that are potentially consistent with the liquidity view: (i) \
small coins have lower prices and higher Amihud illiquidity relative to largecoins, \
(ii) in the cross section, the cryptocurrency size premium is more pro-nounced among \
coins that have high arbitrage costs, and (iii) in the time series,the cryptocurrency \
size premium is larger at times of high cryptocurrency mar-ket volatility. Second, the \
size premium is consistent with a mechanism pro-posed by recent cryptocurrency theories: \
the trade-off between capital gainsand the convenience yield — in equilibrium, the \
convenience yield of larger and more mature cryptocurrencies is higher, and thus \
their capital gain should be lower.Consistent with the prediction that the \
cryptocurrency size premium should be relatively large at times of high demand \
for transactions, we show that the size premium is larger at times of relatively \
high Bitcoin transactions.",
    },
    "mom": {
        "strategy_name": "momentum",
        "rationale": "This strategy is based on the momentum factors. \
Momentum factors are positively correlated with future cryptocurrency cumulative returns. \
The momentum factor reflects the momentum effect of assets. \
Theories of the momentum effect commonly involve behavioral explanations. Many \
studies provide explanations for the momentum effect based on different \
psychological biases. Investor over- and underreaction are both proposed as \
potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
momentum effect is plausibly consistent withthe investor overreaction mechanism. \
After the initial continuation, there is a long-horizon reversal effect. Moreover, \
Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
large and well-known coins. These ﬁndings are in line with the attention-based \
overreaction-induced momentum effect. Consistent with these theories, some studies \
further show that the cryptocurrency momentum effect is more pronounced among coins \
that receive high investor attention and at times of high investor attention.",
    },
}


# common_factor_fine_tuning_dataset = [
#     # Liu et al. (2022) Common Risk Factors in Cryptocurrency
#     {
#         "strategy": "mcap",
#         "description": "log last-day market capitalization in the portfolio formation week.",
#         "monotonicity": "decreasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor mcap. \
# The mcap is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Size Effect:\n"
#         + "The mcap factor reflects the size effect of assets, which are in \
# line with two mechanisms. First, size may proxy for an illiquidity premium. There are \
# three sets of evidence that are potentially consistent with the liquidity view: (i) \
# small coins have lower prices and higher Amihud illiquidity relative to largecoins, \
# (ii) in the cross section, the cryptocurrency size premium is more pro-nounced among \
# coins that have high arbitrage costs, and (iii) in the time series,the cryptocurrency \
# size premium is larger at times of high cryptocurrency mar-ket volatility. Second, the \
# size premium is consistent with a mechanism pro-posed by recent cryptocurrency theories: \
# the trade-off between capital gainsand the convenience yield — in equilibrium, the \
# convenience yield of larger and more mature cryptocurrencies is higher, and thus \
# their capital gain should be lower.Consistent with the prediction that the \
# cryptocurrency size premium should be relatively large at times of high demand \
# for transactions, we show that the size premium is larger at times of relatively \
# high Bitcoin transactions.",
#     },
#     {
#         "strategy": "prc",
#         "description": "log last-day price in the portfolio formation week.",
#         "monotonicity": "decreasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor prc. \
# The prc is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Size Effect:\n"
#         + "The prc factor reflects the size effect of assets, which are in \
# line with two mechanisms. First, size may proxy for an illiquidity premium. There are \
# three sets of evidence that are potentially consistent with the liquidity view: (i) \
# small coins have lower prices and higher Amihud illiquidity relative to largecoins, \
# (ii) in the cross section, the cryptocurrency size premium is more pro-nounced among \
# coins that have high arbitrage costs, and (iii) in the time series,the cryptocurrency \
# size premium is larger at times of high cryptocurrency mar-ket volatility. Second, the \
# size premium is consistent with a mechanism pro-posed by recent cryptocurrency theories: \
# the trade-off between capital gainsand the convenience yield — in equilibrium, the \
# convenience yield of larger and more mature cryptocurrencies is higher, and thus \
# their capital gain should be lower.Consistent with the prediction that the \
# cryptocurrency size premium should be relatively large at times of high demand \
# for transactions, we show that the size premium is larger at times of relatively \
# high Bitcoin transactions.",
#     },
#     {
#         "strategy": "maxdprc",
#         "description": "maximum price of the portfolio formation week.",
#         "monotonicity": "decreasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor maxdprc. \
# The maxdprc is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Size Effect:\n"
#         + "The maxdprc factor reflects the size effect of assets, which are in \
# line with two mechanisms. First, size may proxy for an illiquidity premium. There are \
# three sets of evidence that are potentially consistent with the liquidity view: (i) \
# small coins have lower prices and higher Amihud illiquidity relative to largecoins, \
# (ii) in the cross section, the cryptocurrency size premium is more pro-nounced among \
# coins that have high arbitrage costs, and (iii) in the time series,the cryptocurrency \
# size premium is larger at times of high cryptocurrency mar-ket volatility. Second, the \
# size premium is consistent with a mechanism pro-posed by recent cryptocurrency theories: \
# the trade-off between capital gainsand the convenience yield — in equilibrium, the \
# convenience yield of larger and more mature cryptocurrencies is higher, and thus \
# their capital gain should be lower.Consistent with the prediction that the \
# cryptocurrency size premium should be relatively large at times of high demand \
# for transactions, we show that the size premium is larger at times of relatively \
# high Bitcoin transactions.",
#     },
#     {
#         "strategy": "r_1_0",
#         "description": "past one-week return.",
#         "monotonicity": "increasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor r_1_0. \
# The r_1_0 is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Momentum Effect:\n"
#         + "The r_1_0 factor reflects the momentum effect of assets. \
# Theories of the momentum effect commonly involve behavioral explanations. Many \
# studies provide explanations for the momentum effect based on different \
# psychological biases. Investor over- and underreaction are both proposed as \
# potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
# momentum effect is plausibly consistent withthe investor overreaction mechanism. \
# After the initial continuation, there is a long-horizon reversal effect. Moreover, \
# Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
# large and well-known coins. These ﬁndings are in line with the attention-based \
# overreaction-induced momentum effect. Consistent with these theories, some studies \
# further show that the cryptocurrency momentum effect is more pronounced among coins \
# that receive high investor attention and at times of high investor attention.",
#     },
#     {
#         "strategy": "r_2_0",
#         "description": "past two-week return.",
#         "monotonicity": "increasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor r_2_0. \
# The r_2_0 is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Momentum Effect:\n"
#         + "The r_2_0 factor reflects the momentum effect of assets. \
# Theories of the momentum effect commonly involve behavioral explanations. Many \
# studies provide explanations for the momentum effect based on different \
# psychological biases. Investor over- and underreaction are both proposed as \
# potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
# momentum effect is plausibly consistent withthe investor overreaction mechanism. \
# After the initial continuation, there is a long-horizon reversal effect. Moreover, \
# Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
# large and well-known coins. These ﬁndings are in line with the attention-based \
# overreaction-induced momentum effect. Consistent with these theories, some studies \
# further show that the cryptocurrency momentum effect is more pronounced among coins \
# that receive high investor attention and at times of high investor attention.",
#     },
#     {
#         "strategy": "r_3_0",
#         "description": "past three-week return.",
#         "monotonicity": "increasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor r_3_0. \
# The r_3_0 is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Momentum Effect:\n"
#         + "The r_3_0 factor reflects the momentum effect of assets. \
# Theories of the momentum effect commonly involve behavioral explanations. Many \
# studies provide explanations for the momentum effect based on different \
# psychological biases. Investor over- and underreaction are both proposed as \
# potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
# momentum effect is plausibly consistent withthe investor overreaction mechanism. \
# After the initial continuation, there is a long-horizon reversal effect. Moreover, \
# Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
# large and well-known coins. These ﬁndings are in line with the attention-based \
# overreaction-induced momentum effect. Consistent with these theories, some studies \
# further show that the cryptocurrency momentum effect is more pronounced among coins \
# that receive high investor attention and at times of high investor attention.",
#     },
#     {
#         "strategy": "r_4_0",
#         "description": "past four-week return.",
#         "monotonicity": "increasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor r_4_0. \
# The r_4_0 is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Momentum Effect:\n"
#         + "The r_4_0 factor reflects the momentum effect of assets. \
# Theories of the momentum effect commonly involve behavioral explanations. Many \
# studies provide explanations for the momentum effect based on different \
# psychological biases. Investor over- and underreaction are both proposed as \
# potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
# momentum effect is plausibly consistent withthe investor overreaction mechanism. \
# After the initial continuation, there is a long-horizon reversal effect. Moreover, \
# Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
# large and well-known coins. These ﬁndings are in line with the attention-based \
# overreaction-induced momentum effect. Consistent with these theories, some studies \
# further show that the cryptocurrency momentum effect is more pronounced among coins \
# that receive high investor attention and at times of high investor attention.",
#     },
#     {
#         "strategy": "r_4_1",
#         "description": "past one-to-four-week return.",
#         "monotonicity": "increasing",
#         "rationale": "The cryptocurrency selection strategy is based on the factor r_4_1. \
# The r_4_1 is negatively correlated with future cryptocurrency cumulative returns. \
# Here's a detailed explanation of the rationale:\n"
#         + "Momentum Effect:\n"
#         + "The r_4_1 factor reflects the momentum effect of assets. \
# Theories of the momentum effect commonly involve behavioral explanations. Many \
# studies provide explanations for the momentum effect based on different \
# psychological biases. Investor over- and underreaction are both proposed as \
# potential channels to explain the momentum effect. We ﬁnd that the cryptocurrency \
# momentum effect is plausibly consistent withthe investor overreaction mechanism. \
# After the initial continuation, there is a long-horizon reversal effect. Moreover, \
# Studies ﬁnd that the cryptocurrency momentum effect is markedly stronger among the \
# large and well-known coins. These ﬁndings are in line with the attention-based \
# overreaction-induced momentum effect. Consistent with these theories, some studies \
# further show that the cryptocurrency momentum effect is more pronounced among coins \
# that receive high investor attention and at times of high investor attention.",
#     },
# ]
