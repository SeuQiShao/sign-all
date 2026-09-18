SI4 uses the shared derived plotting data in [`../../fig6/plot_data`](../../fig6/plot_data).
The SI4 renderer reads that directory directly; this folder intentionally does
not duplicate the 71,987-node CSV package.

The shared package contains 119 plotted months from a 120-state SST trajectory:
the first 95 displayed months are training and the last 24 are held-out test
months. Values are in degrees C after inverse normalization, using
`value * 1.3180 + 26.9876` when the physical NPZ arrays are unavailable.

