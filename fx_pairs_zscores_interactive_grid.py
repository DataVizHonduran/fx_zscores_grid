import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import VarianceRatio
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings("ignore", category=InterpolationWarning)

# get_fred_fx: Fetches FX data from FRED, inverts EUR/GBP/AUD/NZD rates to USD base, returns cleaned DataFrame
# safe_adf: Performs Augmented Dickey-Fuller test, returns p-value or NaN if test fails
# safe_kpss: Performs KPSS stationarity test, returns p-value or NaN if test fails
# safe_variance_ratio: Performs variance ratio test, returns p-value or NaN if test fails
# classify_series: Classifies time series as mean reverting, trending, random walk, or insufficient data using statistical tests
# find_cell_coords: Returns rectangle coordinates for highlighting a pair on the heatmap, or None if not found

def get_fred_fx(years=10):
    fx_labels = [
        "USDNOK", "USDSEK", "USDMXN", "USDBRL", "USDZAR",
        "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH",
        "USDJPY", "USDEUR", "USDGBP", "USDCAD", "USDCHF",
        "USDAUD", "USDNZD",
    ]
    
    # Fetch new data from FRED
    datalist = [
        "DEXNOUS", "DEXSDUS", "DEXMXUS", "DEXBZUS", "DEXSFUS",
        "DEXINUS", "DEXKOUS", "DEXTHUS", "DEXSIUS", "DEXCHUS",
        "DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCAUS", "DEXSZUS",
        "DEXUSAL", "DEXUSNZ",
    ]
    
    end_date = datetime.date.today()
    print(f"Fetching FX data from FRED for date range ending: {end_date}")
    start_date = end_date - datetime.timedelta(days=365*years)
    df = data.DataReader(datalist, 'fred', start_date, end_date)
    df.columns = [
        "USDNOK", "USDSEK", "USDMXN", "USDBRL", "USDZAR",
        "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH",
        "USDJPY", "EURUSD", "GBPUSD", "USDCAD", "USDCHF",
        "AUDUSD", "NZDUSD",
    ]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index = pd.to_datetime(df.index)
    
    for old, new in [('EURUSD', 'USDEUR'), ('GBPUSD', 'USDGBP'), 
                     ('AUDUSD', 'USDAUD'), ('NZDUSD', 'USDNZD')]:
        if old in df.columns:
            df[new] = 1 / df[old]
            df = df.drop(old, axis=1)
    
    df = df.bfill()
    
    print(f"Successfully loaded FX data: {df.shape}")
    print(df.tail())
    return df

def safe_adf(series):
    try:
        return adfuller(series, maxlag=10, autolag='AIC')[1]
    except Exception:
        return np.nan

def safe_kpss(series):
    try:
        return kpss(series, regression='c', nlags="auto")[1]
    except Exception:
        return np.nan

def safe_variance_ratio(series):
    try:
        return VarianceRatio(series).pvalue
    except Exception:
        return np.nan

def classify_series(series):
    series = series.dropna()
    if len(series) < 50:
        return 'insufficient_data'
    
    p_adf = safe_adf(series)
    p_kpss = safe_kpss(series)
    if pd.isna(p_adf) or pd.isna(p_kpss):
        return 'test_fail'
    
    if (p_adf < 0.05) and (p_kpss > 0.05):
        return 'mean_reverting'
    
    X = sm.add_constant(np.arange(len(series)))
    model = sm.OLS(series.values, X).fit()
    p_slope = model.pvalues[1] if len(model.pvalues) > 1 else np.nan
    p_vr = safe_variance_ratio(series)
    
    if p_slope < 0.05:
        return 'trending'
    elif (not pd.isna(p_vr)) and (p_vr > 0.05):
        return 'random_walk'
    else:
        return 'random_walk'

def find_cell_coords(pair, x_labels, y_labels):
    try:
        x0 = x_labels.get_loc(pair[1]) - 0.5
        x1 = x_labels.get_loc(pair[1]) + 0.5
        y0 = y_labels.get_loc(pair[0]) - 0.5
        y1 = y_labels.get_loc(pair[0]) + 0.5
        return x0, y0, x1, y1
    except KeyError:
        return None

lookback = 10
df_prices = get_fred_fx(lookback)

window = 100

columns = df_prices.columns.tolist()
ratios = {}
for i, x in enumerate(columns):
    for y in columns[i+1:]:
        ratio_series = df_prices[x] / df_prices[y]
        last_window = ratio_series[-window:]
        mean_ = last_window.mean()
        std_ = last_window.std()
        last_value = last_window.iloc[-1]
        z_score = (last_value - mean_) / std_ if std_ != 0 else 0
        ratios[(x, y)] = {'ratio_series': ratio_series, 'z_score': z_score}

z_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
for (x,y), vals in ratios.items():
    z_matrix.loc[x,y] = vals['z_score']
    z_matrix.loc[y,x] = -vals['z_score']
np.fill_diagonal(z_matrix.values, 0)

results = {
    'mean_reverting': [],
    'trending': [],
    'random_walk': [],
    'insufficient_data': [],
    'test_fail': []
}

for pair, data in ratios.items():
    series = data['ratio_series']
    classification = classify_series(series)
    results[classification].append(pair)

for category, pairs in results.items():
    print(f"{category}: {len(pairs)} pairs")

print("\nTop 5 Mean Reverting Pairs:")
for pair in sorted(results['mean_reverting'])[:5]:
    print(pair)

print("\nTop 5 Trending Pairs:")
for pair in sorted(results['trending'])[:5]:
    print(pair)

print("\nTop 5 Random Walk Pairs:")
for pair in sorted(results['random_walk'])[:5]:
    print(pair)

adf_results = {}
for pair, data in ratios.items():
    series = data['ratio_series'].dropna()
    try:
        pval = adfuller(series)[1]
    except:
        pval = 1
    adf_results[pair] = pval

currency_best_pairs = {}
for currency in z_matrix.index:
    relevant_pairs = [pair for pair in adf_results.keys() if currency in pair]
    if not relevant_pairs:
        continue
    best_pair = min(relevant_pairs, key=lambda p: adf_results[p])
    best_pval = adf_results[best_pair]
    currency_best_pairs[currency] = (best_pair, best_pval)

print("\nTop mean-reverting pairs per currency:")
for c, (pair, pval) in currency_best_pairs.items():
    print(f"{c}: Pair {pair} with ADF p-value = {pval:.4f}")

trend_pvalues = {}
for pair, data in ratios.items():
    series = data['ratio_series'].dropna()
    if len(series) < 20:
        continue
    X = sm.add_constant(np.arange(len(series)))
    model = sm.OLS(series.values, X).fit()
    pval = model.pvalues[1] if len(model.pvalues) > 1 else 1
    trend_pvalues[pair] = pval

currency_best_trending = {}
for currency in z_matrix.index:
    relevant_pairs = [pair for pair in trend_pvalues.keys() if currency in pair]
    if not relevant_pairs:
        continue
    best_pair = min(relevant_pairs, key=lambda p: trend_pvalues[p])
    best_pval = trend_pvalues[best_pair]
    currency_best_trending[currency] = (best_pair, best_pval)

print("\nTop trending pairs per currency:")
for c, (pair, pval) in currency_best_trending.items():
    print(f"{c}: Pair {pair} with trend slope p-value = {pval:.4e}")

mean_reverting_pairs = [pair for pair, pval in currency_best_pairs.values()]
trending_pairs = [pair for pair, pval in currency_best_trending.values()]

fig = make_subplots(
    rows=1, cols=2,
    shared_yaxes=False,
    horizontal_spacing=0.1,
    subplot_titles=(f"Z-Score Heatmap of Price Ratios (last {window} days)", "Price Ratio Over Time")
)

show_low_zs = True
z_masked = z_matrix.copy() if show_low_zs else z_matrix.where(z_matrix.abs() >= 1)

hover_text = z_masked.round(3).astype(str)
hover_text = hover_text.mask(z_masked.isna(), '')

heatmap = go.Heatmap(
    z=z_masked.values,
    x=z_masked.columns,
    y=z_masked.index,
    colorscale='RdYlGn',
    zmid=0,
    colorbar=dict(title='Z-Score'),
    hoverongaps=False,
    text=hover_text.values,
    hoverinfo='text+x+y'
)

default_pair = list(ratios.keys())[0]
default_ratio = ratios[default_pair]['ratio_series']

line = go.Scatter(
    x=default_ratio.index.to_pydatetime(), 
    y=default_ratio.values, 
    mode='lines', 
    name=f'{default_pair[0]}/{default_pair[1]} Ratio'
)

fig.add_trace(heatmap, row=1, col=1)
fig.add_trace(line, row=1, col=2)

shapes = []

for pair in mean_reverting_pairs:
    coords = find_cell_coords(pair, z_masked.columns, z_masked.index)
    if coords:
        x0, y0, x1, y1 = coords
        shapes.append(dict(
            type='rect',
            xref='x',
            yref='y',
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color='black', width=4, dash='solid'),
            fillcolor='rgba(0,0,0,0)',
            layer='above'
        ))

for pair in trending_pairs:
    coords = find_cell_coords(pair, z_masked.columns, z_masked.index)
    if coords:
        x0, y0, x1, y1 = coords
        shapes.append(dict(
            type='rect',
            xref='x',
            yref='y',
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color='black', width=2, dash='dot'),
            fillcolor='rgba(0,0,0,0)',
            layer='above'
        ))

fig.update_layout(
    height=600,
    width=1400,
    title_text=f"FX Price Ratio Explorer - {lookback} Year Lookback (Solid border = mean reverting, Dotted border = trending)",
    shapes=shapes
)

fig.update_xaxes(row=1, col=2)

# Add click interactivity via JavaScript
click_js = """
<script>
var myPlot = document.getElementById('plotly-div');
myPlot.on('plotly_click', function(data){
    if (data.points[0].curveNumber === 0) {  // Only respond to heatmap clicks
        var x_label = data.points[0].x;
        var y_label = data.points[0].y;
        
        // Find the pair in our ratios data
        var pair_key = x_label + '_' + y_label;
        if (!window.ratiosData[pair_key]) {
            pair_key = y_label + '_' + x_label;
        }
        
        if (window.ratiosData[pair_key]) {
            var ratio_data = window.ratiosData[pair_key];
            
            // Check if we should invert
            var recent_mean = ratio_data.y.slice(-100).reduce((a,b) => a+b, 0) / Math.min(100, ratio_data.y.length);
            var invert = recent_mean < 1;
            
            var y_values = invert ? ratio_data.y.map(v => 1/v) : ratio_data.y;
            var title_suffix = invert ? ' (inverted)' : '';
            
            Plotly.restyle('plotly-div', {
                x: [ratio_data.x],
                y: [y_values],
                name: x_label + '/' + y_label + ' Ratio' + title_suffix
            }, [1]);
            
            Plotly.relayout('plotly-div', {
                'annotations[1].text': 'Price Ratio ' + x_label + '/' + y_label + ' Over Time' + title_suffix
            });
        }
    }
});
</script>
"""

# Prepare ratios data for JavaScript
ratios_js_data = {}
for (x, y), data in ratios.items():
    key = f"{x}_{y}"
    ratios_js_data[key] = {
        'x': [d.isoformat() for d in data['ratio_series'].index.to_pydatetime()],
        'y': data['ratio_series'].values.tolist()
    }

# Convert ratios data to JavaScript
import json
ratios_json = json.dumps(ratios_js_data)

# Save as HTML with embedded JavaScript
html_string = fig.to_html(include_plotlyjs='cdn', div_id='plotly-div')

# Insert the ratios data and click handler before </body>
html_parts = html_string.split('</body>')
html_with_js = html_parts[0] + f"""
<script>
window.ratiosData = {ratios_json};
</script>
{click_js}
</body>""" + html_parts[1]

# Save the HTML file
output_file = 'fx_ratio_explorer.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_with_js)

print(f"\nInteractive chart saved to {output_file}")
print("Upload this file to your GitHub repository to display on a webpage.")

warnings.filterwarnings("default", category=InterpolationWarning)
