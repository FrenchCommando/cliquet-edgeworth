import itertools
import numpy as np
import matplotlib.pyplot as plt


def run_montecarlo_cliquet(
        sigma0,
        cap,
        realized,
        n_periods,
        n_mc,
):
    mu = [-0.5 * sigma0 * sigma0] * n_periods
    sigma = [sigma0] * n_periods
    normals = np.random.normal(
        loc=mu,
        scale=sigma,
        size=(n_mc, n_periods),
    )
    r = np.exp(normals)
    r_capped = np.clip(r, a_max=1 + cap, a_min=None)

    def cliquet_from_capped(capped):
        r_sum_capped = np.sum(capped, axis=1)
        return np.clip(r_sum_capped + realized - n_periods, a_min=0, a_max=None)

    r_hit = r > 1 + cap

    d_selected = dict()
    for ref_bool in itertools.product((True, False), repeat=n_periods):
        r_match = np.all(r_hit == ref_bool, axis=1)
        proba_selected = np.mean(r_match)
        r_selected = r_capped[r_match]
        r_cliquet_selected = cliquet_from_capped(capped=r_selected)
        price_cliquet_selected = np.mean(r_cliquet_selected)
        # print(ref_bool, '\t', proba_selected, '\t', price_cliquet_selected)
        d_selected[ref_bool] = dict(
            proba=proba_selected,
            price=price_cliquet_selected,
        )

    r_cliquet = cliquet_from_capped(capped=r_capped)
    price_cliquet = np.mean(r_cliquet)

    return dict(
        r=r,
        r_capped=r_capped,
        r_cliquet=r_cliquet,
        price_cliquet=price_cliquet,
        d=d_selected,
    )


def run_one():
    mc_args = dict(
        sigma0=0.3,
        cap=0.02,
        realized=0.15,
        n_periods=6,
        n_mc=100000,
    )
    out = run_montecarlo_cliquet(**mc_args)
    out_r = out['r']
    out_r_capped = out['r_capped']
    out_r_cliquet = out['r_cliquet']
    n_bins = mc_args['n_mc'] // 100
    plot_returns = False
    if plot_returns:
        plt.figure('Returns')
        for i in range(mc_args['n_periods']):
            plt.hist(out_r[:, i], bins=n_bins)
        plt.show()
    plot_capped_returns = False
    if plot_capped_returns:
        plt.figure('CappedReturns')
        for i in range(mc_args['n_periods']):
            plt.hist(out_r_capped[:, i], bins=n_bins)
        plt.show()
    plot_cliquet = False
    if plot_cliquet:
        plt.figure('CliquetPos')
        out_r_cliquet_pos = out_r_cliquet[out_r_cliquet != 0]
        plt.hist(out_r_cliquet_pos, bins=n_bins)
        plt.show()
    print('PriceCliquet', out['price_cliquet'])


def run_realized_risk():
    mc_args = dict(
        sigma0=0.3,
        cap=0.02,
        # realized=0.15,
        n_periods=6,
        n_mc=100000,
    )
    realized_range = [
        -0.05 + 0.01 * x for x in range(40)
    ]
    cliquet_range = [
        run_montecarlo_cliquet(**dict(
            **mc_args, realized=realized_value,
        ))
        for realized_value in realized_range
    ]
    cliquet_range_mc0 = [
        price_list['price_cliquet']
        for price_list in cliquet_range
    ]
    plot_cliquet_realized = False
    if plot_cliquet_realized:
        plt.figure("Cliquet vs Realized")
        plt.plot(realized_range, cliquet_range_mc0)
        plt.xlabel("Realized")
        plt.ylabel("CliquetPrice")
        plt.title("Cliquet vs Realized")
        plt.savefig("CliquetVsRealized.png")
        plt.show()
    plot_cliquet_realized_d = True
    if plot_cliquet_realized_d:
        # hit_selected = (True, True, True, True, True, True)
        # hit_selected = (False, False, False, False, False, False)
        hit_selected = (False, True, True, True, True, False)
        cliquet_range_d_proba = [
            price_list['d'][hit_selected]['proba']
            for price_list in cliquet_range
        ]
        print(cliquet_range_d_proba)
        # plt.figure("CliquetProba vs Realized")
        # plt.plot(realized_range, cliquet_range_d_proba)
        # plt.xlabel("Realized")
        # plt.ylabel("Proba")
        # plt.show()
        cliquet_range_d_price = [
            price_list['d'][hit_selected]['price']
            for price_list in cliquet_range
        ]
        plt.figure("CliquetProba vs Realized")
        plt.plot(realized_range, cliquet_range_d_price)
        plt.xlabel("Realized")
        plt.ylabel("ConditionalPrice")
        plt.show()
    print("The End")


def main():
    # run_one()
    run_realized_risk()


if __name__ == '__main__':
    main()
