import QuantLib as ql
from time import time

def vanilla_american_binomial_quantlib(S, K, T, r, sigma, q, n, option_type):
    """
    American vanilla option pricing using QuantLib's BinomialVanillaEngine.
    """

    opt_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put

    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    valuation_date = ql.Date.todaysDate()
    maturity_date = valuation_date + int(T * 365)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, r, day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, q, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, sigma, day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, q_handle, r_handle, vol_handle)

    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.AmericanExercise(valuation_date, maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    engine = ql.BinomialVanillaEngine(bsm_process, "crr", n)
    option.setPricingEngine(engine)

    return option.NPV()

start = time()
print(vanilla_american_binomial_quantlib(100, 100, 1, 0.03, 0.2, 0.015, 100000, "put"))
end = time() - start
print(f"Time taken: {end * 1000} ms")