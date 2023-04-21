import numpy as np
import matplotlib.pyplot as plt
from montecarlo_cliquet import run_montecarlo_cliquet
from approximation_cliquet import run_approximation_cliquet


def compare_cliquet_realized_risk():
    realized_range = [
        -0.3 + 0.01 * x for x in range(60)
    ]
    common_args = dict(
        # sigma0=0.5,
        sigma0=0.05,
        cap=0.02,
        n_periods=6,
    )
    cliquet_range_mc = [
        run_montecarlo_cliquet(**dict(
            **common_args, realized=realized_value, n_mc=100000,
        ))
        for realized_value in realized_range
    ]
    cliquet_range_mc0 = np.array([
        price_list['price_cliquet']
        for price_list in cliquet_range_mc
    ])
    cliquet_range_approx = [
        run_approximation_cliquet(**dict(
            **common_args, realized=realized_value,
        ))
        for realized_value in realized_range
    ]
    cliquet_range_approx0 = np.array([
        price_list['price_cliquet0']
        for price_list in cliquet_range_approx
    ])
    cliquet_range_approx3 = np.array([
        price_list['price_cliquet3']
        for price_list in cliquet_range_approx
    ])
    cliquet_range_approx4 = np.array([
        price_list['price_cliquet4']
        for price_list in cliquet_range_approx
    ])
    cliquet_range_approx_last = np.array([
        price_list['price_cliquet_last']
        for price_list in cliquet_range_approx
    ])
    plot_cliquet_realized = True
    if plot_cliquet_realized:
        plt.figure("Cliquet vs Realized")
        plt.plot(realized_range, cliquet_range_mc0, label="MC")
        plt.plot(realized_range, cliquet_range_approx0, label="Approx0",
                 linestyle='dashed', dashes=(2, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx3, label="Approx3",
                 linestyle='dashed', dashes=(3, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx4, label="Approx4",
                 linestyle='dashed', dashes=(4, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx_last, label="ApproxLast",
                 linestyle='dashed', dashes=(4, 1), linewidth=1)
        plt.legend()
        plt.xlabel("Realized")
        plt.ylabel("CliquetPrice")
        plt.title(f"Cliquet vs Realized {common_args}")
        plt.grid(visible=True)
        plt.savefig("CliquetVsRealized.png")
        plt.figure("Cliquet vs Realized - Diff")
        plt.plot(realized_range, cliquet_range_mc0 - cliquet_range_mc0)
        plt.plot(realized_range, cliquet_range_approx0 - cliquet_range_mc0, label="Approx0 - MC",
                 linestyle='dashed', dashes=(2, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx3 - cliquet_range_mc0, label="Approx3 - MC",
                 linestyle='dashed', dashes=(3, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx4 - cliquet_range_mc0, label="Approx4 - MC",
                 linestyle='dashed', dashes=(4, 1), linewidth=1)
        plt.plot(realized_range, cliquet_range_approx_last - cliquet_range_mc0, label="ApproxLast - MC",
                 linestyle='dashed', dashes=(4, 1), linewidth=1)
        plt.legend()
        plt.xlabel("Realized")
        plt.ylabel("CliquetPrice - diff")
        plt.title(f"Cliquet vs Realized - diff {common_args}")
        plt.grid(visible=True)
        plt.savefig("CliquetVsRealizedDiff.png")
        # plt.show()

    plot_cliquet_realized_d = True
    if plot_cliquet_realized_d:
        for hit_selected in [
            tuple(True if j < i else False for j in range(common_args['n_periods']))
            for i in range(common_args['n_periods'] + 1)
        ]:
            cliquet_range_d_proba_mc = [
                price_list['d'][hit_selected]['proba']
                for price_list in cliquet_range_mc
            ]
            cliquet_range_d_proba_approx = [
                price_list['d'][hit_selected]['proba']
                for price_list in cliquet_range_approx
            ]
            proba_value_mc = np.mean(cliquet_range_d_proba_mc)
            proba_value_approx = np.mean(cliquet_range_d_proba_approx)
            print("Hit sequence", hit_selected)
            print("Conditional Price MC", proba_value_mc)
            print("Conditional Price Approx", proba_value_approx)

            cliquet_range_d_price_mc = [
                price_list['d'][hit_selected]['price']
                for price_list in cliquet_range_mc
            ]
            cliquet_range_d_price_approx0 = [
                price_list['d'][hit_selected]['price0']
                for price_list in cliquet_range_approx
            ]
            cliquet_range_d_price_approx3 = [
                price_list['d'][hit_selected]['price3']
                for price_list in cliquet_range_approx
            ]
            cliquet_range_d_price_approx4 = [
                price_list['d'][hit_selected]['price4']
                for price_list in cliquet_range_approx
            ]
            cliquet_range_d_price_approx_last = [
                price_list['d'][hit_selected]['price_last']
                for price_list in cliquet_range_approx
            ]

            hit_code = ''.join('T' if b else 'F' for b in hit_selected)
            plt.figure(f"CliquetProba vs Realized {hit_code}")
            plt.plot(realized_range, cliquet_range_d_price_mc, label='ConditionalPriceMC')
            plt.plot(realized_range, cliquet_range_d_price_approx0,
                     linestyle='dashed', dashes=(4, 1), linewidth=1,
                     label='ConditionalPriceApprox0'
                     )
            plt.plot(realized_range, cliquet_range_d_price_approx3,
                     linestyle='dashed', dashes=(3, 1), linewidth=1,
                     label='ConditionalPriceApprox3'
                     )
            plt.plot(realized_range, cliquet_range_d_price_approx4,
                     linestyle='dashed', dashes=(3, 2), linewidth=1,
                     label='ConditionalPriceApprox4'
                     )
            plt.plot(realized_range, cliquet_range_d_price_approx_last,
                     linestyle='dashed', dashes=(5, 2), linewidth=1,
                     label='ConditionalPriceApproxLast'
                     )
            plt.legend()
            plt.xlabel("Realized")
            plt.ylabel("ConditionalPrice")
            plt.title(f"{hit_selected} proba {proba_value_mc}")
            plt.grid(visible=True)
            plt.savefig(f"Conditional{hit_code}.png")
        plt.show()
    print("The End - Cliquet compare")


def main():
    compare_cliquet_realized_risk()


if __name__ == '__main__':
    main()
