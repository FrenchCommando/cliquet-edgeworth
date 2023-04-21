import itertools
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def h1(x):
    return x


def h2(x):
    return x * x - 1


def h3(x):
    return (x * x - 3) * x


def h4(x):
    return (x * x - 6) * x * x + 3


def h5(x):
    return ((x * x - 10) * x * x + 15) * x


def run_approximation_cliquet(
        sigma0,
        cap,
        realized,
        n_periods,
):
    p_m1 = norm.cdf(np.log(1 + cap) / sigma0 + 0.5 * sigma0)
    p = 1 - p_m1
    m1_top = norm.cdf(np.log(1 + cap) / sigma0 - 0.5 * sigma0)
    m1 = m1_top / p_m1
    m2_top = norm.cdf(np.log(1 + cap) / sigma0 - 1.5 * sigma0)
    m3_top = norm.cdf(np.log(1 + cap) / sigma0 - 2.5 * sigma0)
    m4_top = norm.cdf(np.log(1 + cap) / sigma0 - 3.5 * sigma0)
    m5_top = norm.cdf(np.log(1 + cap) / sigma0 - 4.5 * sigma0)
    m2 = m2_top / p_m1 * np.exp(0.5 * sigma0 * sigma0 * 2 * 1)
    m3 = m3_top / p_m1 * np.exp(0.5 * sigma0 * sigma0 * 3 * 2)
    m4 = m4_top / p_m1 * np.exp(0.5 * sigma0 * sigma0 * 4 * 3)
    m5 = m5_top / p_m1 * np.exp(0.5 * sigma0 * sigma0 * 5 * 4)
    k2 = m2 - m1 * m1
    k3 = m3 - 3 * m2 * m1 + 2 * m1 * m1 * m1
    k4 = m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1 * m1 * m1 * m1
    k5 = m5 - 5 * m4 * m1 - 10 * m3 * m2 + 20 * m3 * m1 * m1 \
        + 30 * m2 * m2 * m1 - 60 * m2 * m1 * m1 * m1 + 24 * m1 * m1 * m1 * m1 * m1

    # this is for the case, not hit == 1
    k_one = 1 - cap * (n_periods - 1) - realized
    if k_one <= 0:
        gk = 0
        gk_one = 0
    else:
        gk = norm.cdf(np.log(k_one) / sigma0 + 0.5 * sigma0)
        gk_one = norm.cdf(np.log(k_one) / sigma0 - 0.5 * sigma0)
    conditional_price_one = (m1_top - gk_one) / p_m1 - k_one * (p_m1 - gk) / p_m1

    proba_sum = 0
    cliquet_price0 = 0
    cliquet_price3 = 0
    cliquet_price4 = 0
    cliquet_price_last = 0
    d_selected = dict()
    for ref_bool in itertools.product((True, False), repeat=n_periods):
        n_hit = sum(ref_bool)
        n_not_hit = n_periods - sum(ref_bool)
        hit_proba = (p ** n_hit) * (p_m1 ** n_not_hit)

        if n_not_hit == 0:
            hit_price = max(0, n_hit * cap + realized)
            d_selected[ref_bool] = dict(
                proba=hit_proba,
                price0=hit_price,
                price3=hit_price,
                price4=hit_price,
                price_last=hit_price,
            )
        else:
            mu = m1 * n_not_hit
            sigma = np.sqrt(k2 * n_not_hit)
            # print(ref_bool, mu, sigma)
            l3 = k3 * n_not_hit / (sigma ** 3)
            l4 = k4 * n_not_hit / (sigma ** 4)
            l5 = k5 * n_not_hit / (sigma ** 5)

            a = n_not_hit - cap * n_hit - realized
            b = (1 + cap) * n_not_hit
            za = (a - mu) / sigma
            zb = (b - mu) / sigma

            i0 = sigma * (norm.pdf(za) - norm.pdf(zb)) + (mu - a) * (norm.cdf(zb) - norm.cdf(za))
            i3 = \
                sigma * (norm.pdf(zb) * h1(x=zb) - norm.pdf(za) * h1(x=za)) \
                - (b - a) * norm.pdf(zb) * h2(x=zb)
            i4 = \
                sigma * (norm.pdf(zb) * h2(x=zb) - norm.pdf(za) * h2(x=za)) \
                - (b - a) * norm.pdf(zb) * h3(x=zb)
            i6 = \
                sigma * (norm.pdf(zb) * h4(x=zb) - norm.pdf(za) * h4(x=za)) \
                - (b - a) * norm.pdf(zb) * h5(x=zb)

            def bound(x):
                return max(0, min(x, n_periods * cap + realized))

            hit_price0 = bound(x=i0)
            hit_price3 = bound(x=i0 + l3 / 6 * i3)
            hit_price4 = bound(x=i0 + l3 / 6 * i3 + l4 / 24 * i4 + l3 * l3 / 72 * i6)
            hit_price_last = bound(x=i0 + l3 / 6 * i3 + l4 / 24 * i4 + l3 * l3 / 72 * i6)
            if n_not_hit == 1:
                hit_price_last = bound(x=conditional_price_one)
            proba_sum += hit_proba
            cliquet_price0 += hit_proba * hit_price0
            cliquet_price3 += hit_proba * hit_price3
            cliquet_price4 += hit_proba * hit_price4
            cliquet_price_last += hit_proba * hit_price_last
            d_selected[ref_bool] = dict(
                proba=hit_proba,
                price0=hit_price0,
                price3=hit_price3,
                price4=hit_price4,
                price_last=hit_price_last,
            )
    # print('Sum proba', proba_sum)
    return dict(
        price_cliquet0=cliquet_price0,
        price_cliquet3=cliquet_price3,
        price_cliquet4=cliquet_price4,
        price_cliquet_last=cliquet_price_last,
        d=d_selected,
    )


def run_approximation_risk():
    approx_args = dict(
        sigma0=0.3,
        cap=0.02,
        # realized=0.15,
        n_periods=6,
    )
    realized_range = [
        -0.05 + 0.01 * x for x in range(40)
    ]
    cliquet_range = [
        run_approximation_cliquet(**dict(
            **approx_args, realized=realized_value,
        ))['price_cliquet4']
        for realized_value in realized_range
    ]
    plot_cliquet_realized = True
    if plot_cliquet_realized:
        plt.figure("Cliquet vs Realized")
        plt.plot(realized_range, cliquet_range)
        plt.xlabel("Realized")
        plt.ylabel("CliquetPrice")
        plt.title("Cliquet vs Realized")
        plt.savefig("CliquetVsRealized.png")
        plt.show()
    print("The End -- cliquet approximation realized")


def main():
    run_approximation_risk()


if __name__ == '__main__':
    main()
