# Pandas & Matplotlib Comprehensive Study Guide
# This guide covers essential pandas and matplotlib functions with examples
# Each section can be run independently - uncomment the sections you want to test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: PANDAS BASICS
# =============================================================================

# Creating DataFrames and Series
def pandas_basics():
    print("=== PANDAS BASICS ===")
    
    # Creating a DataFrame from dictionary
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'city': ['NYC', 'LA', 'Chicago', 'Miami'],
        'salary': [70000, 80000, 90000, 75000]
    }
    df = pd.DataFrame(data)
    print("Basic DataFrame:")
    print(df)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame info:")
    print(df.info())
    
    # Series creation and basic operations
    ages = pd.Series([25, 30, 35, 28], name='ages')
    print(f"\nSeries mean: {ages.mean()}")
    print(f"Series median: {ages.median()}")
    
    return df

# Uncomment to run:
# df_basic = pandas_basics()

# =============================================================================
# SECTION 2: DATA SELECTION AND FILTERING
# =============================================================================

def data_selection():
    print("\n=== DATA SELECTION AND FILTERING ===")
    
    # Create sample data
    df = pd.DataFrame({
        'A': range(1, 11),
        'B': range(10, 20),
        'C': ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 'x'],
        'D': np.random.randn(10)
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Column selection
    print("\n--- Column Selection ---")
    print("Single column (Series):")
    print(df['A'].head(3))
    
    print("\nMultiple columns (DataFrame):")
    print(df[['A', 'C']].head(3))
    
    # Row selection with loc and iloc
    print("\n--- Row Selection ---")
    print("Using iloc (position-based):")
    print(df.iloc[0:3])  # First 3 rows
    
    print("\nUsing loc (label-based):")
    print(df.loc[0:2, 'A':'C'])  # Rows 0-2, columns A-C
    
    # Boolean filtering
    print("\n--- Boolean Filtering ---")
    mask = df['A'] > 5
    print(f"Rows where A > 5:")
    print(df[mask])
    
    # Multiple conditions
    complex_mask = (df['A'] > 3) & (df['C'] == 'x')
    print(f"\nRows where A > 3 AND C == 'x':")
    print(df[complex_mask])
    
    # Query method (alternative to boolean indexing)
    print(f"\nUsing query method:")
    print(df.query("A > 5 and C == 'x'"))

# Uncomment to run:
# data_selection()

# =============================================================================
# SECTION 3: DATETIME OPERATIONS
# =============================================================================

def datetime_operations():
    print("\n=== DATETIME OPERATIONS ===")
    
    # Creating datetime data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()
    df = pd.DataFrame({'date': dates, 'value': values})
    
    print("DateTime DataFrame:")
    print(df.head())
    
    # Set date as index for time series operations
    df.set_index('date', inplace=True)
    
    # DateTime properties
    print("\n--- DateTime Properties ---")
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.day_name()
    df['quarter'] = df.index.quarter
    
    print("DataFrame with datetime components:")
    print(df.head())
    
    # Date filtering
    print("\n--- Date Filtering ---")
    january_data = df['2023-01']  # All January data
    print(f"January 2023 data (first 5 rows):")
    print(january_data.head())
    
    # Date range filtering
    date_range = df['2023-01-15':'2023-01-25']
    print(f"\nData from Jan 15-25:")
    print(date_range)
    
    # Working with different datetime formats
    print("\n--- DateTime Parsing ---")
    date_strings = ['2023/01/15', '2023-02-20', '15/03/2023']
    parsed_dates = pd.to_datetime(date_strings, infer_datetime_format=True)
    print(f"Parsed dates: {parsed_dates}")
    
    return df

# Uncomment to run:
# df_datetime = datetime_operations()

# =============================================================================
# SECTION 4: RESAMPLING AND TIME SERIES
# =============================================================================

def resampling_operations():
    print("\n=== RESAMPLING OPERATIONS ===")
    
    # Create time series data with higher frequency
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)
    values = np.random.randn(365).cumsum() + 100
    df = pd.DataFrame({'value': values}, index=dates)
    
    print("Original daily data (first 10 days):")
    print(df.head(10))
    
    # Downsampling (higher frequency to lower frequency)
    print("\n--- Downsampling ---")
    
    # Resample to weekly mean
    weekly_mean = df.resample('W').mean()
    print("Weekly mean (first 5 weeks):")
    print(weekly_mean.head())
    
    # Resample to monthly with different aggregations
    monthly_agg = df.resample('M').agg({
        'value': ['mean', 'max', 'min', 'std']
    })
    print("\nMonthly aggregations (first 3 months):")
    print(monthly_agg.head(3))
    
    # Resample with custom functions
    quarterly = df.resample('Q').agg({
        'value': [
            ('avg', 'mean'),
            ('total', 'sum'),
            ('volatility', lambda x: x.std())
        ]
    })
    print("\nQuarterly custom aggregations:")
    print(quarterly)
    
    # Upsampling (lower frequency to higher frequency)
    print("\n--- Upsampling ---")
    monthly_data = df.resample('M').mean()
    
    # Forward fill
    daily_ffill = monthly_data.resample('D').ffill()
    print("Upsampled with forward fill (first 10 days):")
    print(daily_ffill.head(10))
    
    # Interpolation
    daily_interp = monthly_data.resample('D').interpolate()
    print("\nUpsampled with interpolation (first 10 days):")
    print(daily_interp.head(10))
    
    return df, weekly_mean, monthly_agg

# Uncomment to run:
# df_ts, weekly, monthly = resampling_operations()

# =============================================================================
# SECTION 5: GROUPBY OPERATIONS
# =============================================================================

def groupby_operations():
    print("\n=== GROUPBY OPERATIONS ===")
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'subcategory': np.random.choice(['X', 'Y'], 100),
        'value1': np.random.randn(100) * 10 + 50,
        'value2': np.random.randint(1, 100, 100),
        'date': pd.date_range('2023-01-01', periods=100)
    })
    
    print("Sample data:")
    print(df.head())
    
    # Basic groupby
    print("\n--- Basic GroupBy ---")
    grouped = df.groupby('category')['value1'].mean()
    print("Mean value1 by category:")
    print(grouped)
    
    # Multiple aggregations
    print("\n--- Multiple Aggregations ---")
    agg_result = df.groupby('category').agg({
        'value1': ['mean', 'std', 'count'],
        'value2': ['sum', 'max']
    })
    print("Multiple aggregations by category:")
    print(agg_result)
    
    # Custom aggregation functions
    print("\n--- Custom Aggregations ---")
    custom_agg = df.groupby('category')['value1'].agg([
        ('mean', 'mean'),
        ('range', lambda x: x.max() - x.min()),
        ('cv', lambda x: x.std() / x.mean())  # Coefficient of variation
    ])
    print("Custom aggregations:")
    print(custom_agg)
    
    # Multiple column grouping
    print("\n--- Multiple Column GroupBy ---")
    multi_group = df.groupby(['category', 'subcategory'])['value1'].mean().unstack()
    print("Mean value1 by category and subcategory:")
    print(multi_group)
    
    # Transform operations
    print("\n--- Transform Operations ---")
    df['value1_zscore'] = df.groupby('category')['value1'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print("Data with z-scores by category (first 10 rows):")
    print(df[['category', 'value1', 'value1_zscore']].head(10))

# Uncomment to run:
# groupby_operations()

# =============================================================================
# SECTION 6: DATA CLEANING AND MANIPULATION
# =============================================================================

def data_cleaning():
    print("\n=== DATA CLEANING AND MANIPULATION ===")
    
    # Create messy data
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
        'age': [25, None, 35, 28, 45],
        'salary': [70000, 80000, None, 75000, 95000],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance']
    })
    
    print("Original messy data:")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handling missing values
    print("\n--- Handling Missing Values ---")
    
    # Drop rows with any missing values
    df_dropped = df.dropna()
    print("After dropping rows with any NaN:")
    print(df_dropped)
    
    # Fill missing values
    df_filled = df.copy()
    df_filled['name'].fillna('Unknown', inplace=True)
    df_filled['age'].fillna(df_filled['age'].mean(), inplace=True)
    df_filled['salary'].fillna(df_filled['salary'].median(), inplace=True)
    
    print("\nAfter filling missing values:")
    print(df_filled)
    
    # Data type conversion
    print("\n--- Data Type Conversion ---")
    print(f"Original dtypes:\n{df_filled.dtypes}")
    
    df_filled['age'] = df_filled['age'].astype(int)
    df_filled['salary'] = df_filled['salary'].astype(int)
    
    print(f"\nAfter conversion:\n{df_filled.dtypes}")
    
    # String operations
    print("\n--- String Operations ---")
    df_filled['name_upper'] = df_filled['name'].str.upper()
    df_filled['name_length'] = df_filled['name'].str.len()
    df_filled['dept_contains_i'] = df_filled['department'].str.contains('I', na=False)
    
    print("After string operations:")
    print(df_filled)
    
    return df_filled

# Uncomment to run:
# df_clean = data_cleaning()

# =============================================================================
# SECTION 7: MERGING AND JOINING
# =============================================================================

def merging_joining():
    print("\n=== MERGING AND JOINING ===")
    
    # Create sample datasets
    employees = pd.DataFrame({
        'emp_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'dept_id': [10, 20, 10, 30]
    })
    
    departments = pd.DataFrame({
        'dept_id': [10, 20, 30, 40],
        'dept_name': ['Engineering', 'Marketing', 'Sales', 'HR'],
        'budget': [100000, 80000, 60000, 50000]
    })
    
    print("Employees:")
    print(employees)
    print("\nDepartments:")
    print(departments)
    
    # Inner join
    print("\n--- Inner Join ---")
    inner_join = pd.merge(employees, departments, on='dept_id', how='inner')
    print(inner_join)
    
    # Left join
    print("\n--- Left Join ---")
    left_join = pd.merge(employees, departments, on='dept_id', how='left')
    print(left_join)
    
    # Right join
    print("\n--- Right Join ---")
    right_join = pd.merge(employees, departments, on='dept_id', how='right')
    print(right_join)
    
    # Outer join
    print("\n--- Outer Join ---")
    outer_join = pd.merge(employees, departments, on='dept_id', how='outer')
    print(outer_join)
    
    # Concatenation
    print("\n--- Concatenation ---")
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    
    # Vertical concatenation
    vertical_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
    print("Vertical concatenation:")
    print(vertical_concat)
    
    # Horizontal concatenation
    horizontal_concat = pd.concat([df1, df2], axis=1)
    print("\nHorizontal concatenation:")
    print(horizontal_concat)

# Uncomment to run:
# merging_joining()

# =============================================================================
# SECTION 8: MATPLOTLIB BASICS
# =============================================================================

def matplotlib_basics():
    print("\n=== MATPLOTLIB BASICS ===")
    
    # Basic line plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', linewidth=2)
    plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Basic Line Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = x_scatter + np.random.randn(100) * 0.5
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_scatter, y_scatter, alpha=0.6, c=y_scatter, cmap='viridis')
    plt.colorbar(label='y value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot with Color Mapping')
    plt.show()
    
    # Subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Line plot
    axes[0, 0].plot(x, y1)
    axes[0, 0].set_title('Sin Function')
    axes[0, 0].grid(True)
    
    # Plot 2: Histogram
    axes[0, 1].hist(np.random.randn(1000), bins=30, alpha=0.7)
    axes[0, 1].set_title('Histogram')
    
    # Plot 3: Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [20, 35, 30, 35]
    axes[1, 0].bar(categories, values)
    axes[1, 0].set_title('Bar Plot')
    
    # Plot 4: Box plot
    data = [np.random.randn(100), np.random.randn(100) + 1, np.random.randn(100) - 1]
    axes[1, 1].boxplot(data)
    axes[1, 1].set_title('Box Plot')
    
    plt.tight_layout()
    plt.show()

# Uncomment to run:
# matplotlib_basics()

# =============================================================================
# SECTION 9: ADVANCED MATPLOTLIB
# =============================================================================

def advanced_matplotlib():
    print("\n=== ADVANCED MATPLOTLIB ===")
    
    # Time series plotting with pandas integration
    dates = pd.date_range('2023-01-01', periods=365)
    ts_data = pd.DataFrame({
        'value1': np.random.randn(365).cumsum(),
        'value2': np.random.randn(365).cumsum(),
        'value3': np.random.randn(365).cumsum()
    }, index=dates)
    
    # Multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value 1', color=color)
    ax1.plot(ts_data.index, ts_data['value1'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Value 2', color=color)
    ax2.plot(ts_data.index, ts_data['value2'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Time Series with Dual Y-Axes')
    fig.tight_layout()
    plt.show()
    
    # Heatmap
    corr_matrix = ts_data.corr()
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Heatmap')
    
    # Add correlation values to heatmap
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.show()
    
    # Customized styling
    plt.style.use('seaborn-v0_8')  # Use seaborn style
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create some sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 50)
    
    ax.plot(x, y1, 'o-', label='Data 1', markersize=4, alpha=0.8)
    ax.plot(x, y2, 's-', label='Data 2', markersize=4, alpha=0.8)
    ax.fill_between(x, y1, alpha=0.3)
    ax.fill_between(x, y2, alpha=0.3)
    
    ax.set_xlabel('X Value', fontsize=12)
    ax.set_ylabel('Y Value', fontsize=12)
    ax.set_title('Customized Plot with Fill', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Uncomment to run:
# advanced_matplotlib()

# =============================================================================
# SECTION 10: PANDAS + MATPLOTLIB INTEGRATION
# =============================================================================

def pandas_matplotlib_integration():
    print("\n=== PANDAS + MATPLOTLIB INTEGRATION ===")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365)
    df = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 100,
        'marketing_spend': np.random.normal(100, 20, 365),
        'temperature': np.random.normal(20, 10, 365) + 15 * np.sin(np.arange(365) * 2 * np.pi / 365),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365)
    }, index=dates)
    
    # Add some derived columns
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_week'] = df.index.day_name()
    
    print("Sample business data:")
    print(df.head())
    
    # Using pandas plotting directly
    print("\n--- Pandas Built-in Plotting ---")
    
    # Time series plot
    df[['sales', 'marketing_spend']].plot(figsize=(12, 6), secondary_y='marketing_spend')
    plt.title('Sales and Marketing Spend Over Time')
    plt.show()
    
    # Monthly aggregation and plotting
    monthly_data = df.groupby('month').agg({
        'sales': 'mean',
        'marketing_spend': 'mean',
        'temperature': 'mean'
    })
    
    monthly_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Monthly Averages')
    plt.xlabel('Month')
    plt.ylabel('Average Values')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Box plots by category
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    df.boxplot(column='sales', by='region', ax=axes[0])
    axes[0].set_title('Sales by Region')
    axes[0].set_xlabel('Region')
    
    # Day of week analysis
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df.groupby('day_of_week')['sales'].mean().reindex(day_order)
    df_day.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Sales by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis with visualization
    correlation_matrix = df[['sales', 'marketing_spend', 'temperature']].corr()
    
    # Create a more sophisticated heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.columns)
    
    # Add correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return df

# Uncomment to run:
# df_business = pandas_matplotlib_integration()

# =============================================================================
# SECTION 11: PERFORMANCE TIPS AND BEST PRACTICES
# =============================================================================

def performance_tips():
    print("\n=== PERFORMANCE TIPS AND BEST PRACTICES ===")
    
    # Vectorization vs loops
    print("--- Vectorization Example ---")
    
    # Create large dataset
    n = 1000000
    df = pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n)
    })
    
    import time
    
    # Bad: Using loops
    start_time = time.time()
    result_loop = []
    for i in range(min(10000, len(df))):  # Only first 10k for demo
        result_loop.append(df.iloc[i]['A'] * df.iloc[i]['B'])
    loop_time = time.time() - start_time
    
    # Good: Using vectorization
    start_time = time.time()
    result_vectorized = (df['A'] * df['B']).head(10000)
    vectorized_time = time.time() - start_time
    
    print(f"Loop time (10k rows): {loop_time:.4f} seconds")
    print(f"Vectorized time (10k rows): {vectorized_time:.4f} seconds")
    print(f"Speedup: {loop_time/vectorized_time:.1f}x faster")
    
    # Memory optimization
    print("\n--- Memory Optimization ---")
    
    # Check memory usage
    print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Optimize data types
    df_optimized = df.copy()
    
    # Convert to more efficient data types if possible
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Efficient filtering
    print("\n--- Efficient Filtering ---")
    
    # Use query for complex conditions
    condition = "A > 0 and B < 0"
    
    # Method 1: Boolean indexing
    start_time = time.time()
    result1 = df[(df['A'] > 0) & (df['B'] < 0)]
    time1 = time.time() - start_time
    
    # Method 2: Query method
    start_time = time.time()
    result2 = df.query(condition)
    time2 = time.time() - start_time
    
    print(f"Boolean indexing time: {time1:.4f} seconds")
    print(f"Query method time: {time2:.4f} seconds")
    
    # Best practices summary
    print("\n--- Best Practices Summary ---")
    practices = [
        "1. Use vectorized operations instead of loops",
        "2. Optimize data types (use appropriate int/float sizes)",
        "3. Use categorical data type for repeated strings",
        "4. Use query() for complex filtering conditions",
        "5. Avoid chained indexing (use .loc instead)",
        "6. Use inplace=True for memory efficiency when appropriate",
        "7. Use chunking for very large datasets",
        "8. Profile your code to identify bottlenecks"
    ]
    
    for practice in practices:
        print(practice)

# Uncomment to run:
# performance_tips()

# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================

if __name__ == "__main__":
    print("Pandas & Matplotlib Study Guide")
    print("=" * 50)
    print("Uncomment any section above to run specific examples:")
    print("- pandas_basics()")
    print("- data_selection()")
    print("- datetime_operations()")
    print("- resampling_operations()")
    print("- groupby_operations()")
    print("- data_cleaning()")
    print("- merging_joining()")
    print("- matplotlib_basics()")
    print("- advanced_matplotlib()")
    print("- pandas_matplotlib_integration()")
    print("- performance_tips()")
    
    # Uncomment the sections you want to run:
    
    # Basic pandas operations
    # df_basic = pandas_basics()
    # data_selection()
    
    # Time series and datetime
    # df_datetime = datetime_operations()
    # df_ts, weekly, monthly = resampling_operations()
    
    # Data manipulation
    # groupby_operations()
    # df_clean = data_cleaning()
    # merging_joining()
    
    # Visualization
    # matplotlib_basics()
    # advanced_matplotlib()
    # df_business = pandas_matplotlib_integration()
    
    # Performance
    # performance_tips()
    
    print("\nStudy guide loaded successfully!")
    print("Uncomment sections to run examples and see outputs!")

# =============================================================================
# SECTION 12: PIVOT TABLES AND CROSSTABS
# =============================================================================

def pivot_operations():
    print("\n=== PIVOT TABLES AND CROSSTABS ===")
    
    # Create sample sales data
    np.random.seed(42)
    data = []
    products = ['Laptop', 'Phone', 'Tablet', 'Watch']
    regions = ['North', 'South', 'East', 'West']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for _ in range(200):
        data.append({
            'Product': np.random.choice(products),
            'Region': np.random.choice(regions),
            'Quarter': np.random.choice(quarters),
            'Sales': np.random.randint(1000, 10000),
            'Units': np.random.randint(10, 100),
            'Profit': np.random.randint(100, 2000)
        })
    
    df = pd.DataFrame(data)
    print("Sample sales data:")
    print(df.head(10))
    
    # Basic pivot table
    print("\n--- Basic Pivot Table ---")
    pivot_basic = df.pivot_table(
        values='Sales', 
        index='Product', 
        columns='Region', 
        aggfunc='mean'
    )
    print("Average sales by Product and Region:")
    print(pivot_basic)
    
    # Multi-level pivot table
    print("\n--- Multi-level Pivot Table ---")
    pivot_multi = df.pivot_table(
        values=['Sales', 'Profit'], 
        index=['Product', 'Quarter'], 
        columns='Region',
        aggfunc={'Sales': 'sum', 'Profit': 'mean'},
        fill_value=0,
        margins=True  # Add totals
    )
    print("Multi-level pivot with totals:")
    print(pivot_multi.head(10))
    
    # Crosstab
    print("\n--- Crosstab ---")
    crosstab = pd.crosstab(
        df['Product'], 
        df['Region'], 
        values=df['Units'], 
        aggfunc='sum',
        margins=True
    )
    print("Units sold crosstab:")
    print(crosstab)
    
    # Pivot with multiple aggregation functions
    print("\n--- Multiple Aggregation Functions ---")
    pivot_agg = df.pivot_table(
        values='Sales',
        index='Product',
        columns='Quarter',
        aggfunc=['mean', 'sum', 'count'],
        fill_value=0
    )
    print("Sales with multiple aggregations:")
    print(pivot_agg)
    
    return df, pivot_basic

# Uncomment to run:
# df_sales, pivot_sales = pivot_operations()

# =============================================================================
# SECTION 13: ADVANCED INDEXING AND MULTIINDEX
# =============================================================================

def multiindex_operations():
    print("\n=== MULTIINDEX OPERATIONS ===")
    
    # Create MultiIndex DataFrame
    arrays = [
        ['A', 'A', 'B', 'B', 'C', 'C'],
        ['X', 'Y', 'X', 'Y', 'X', 'Y']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
    df = pd.DataFrame(np.random.randn(6, 4), index=index, columns=['col1', 'col2', 'col3', 'col4'])
    
    print("MultiIndex DataFrame:")
    print(df)
    
    # Accessing MultiIndex data
    print("\n--- Accessing MultiIndex Data ---")
    print("Level 0 selection (A):")
    print(df.loc['A'])
    
    print("\nSpecific index selection (A, X):")
    print(df.loc[('A', 'X')])
    
    print("\nCross-section (xs) - all X's:")
    print(df.xs('X', level='second'))
    
    # Stacking and unstacking
    print("\n--- Stacking and Unstacking ---")
    print("Original shape:", df.shape)
    
    # Stack columns to index
    df_stacked = df.stack()
    print("After stacking:")
    print(df_stacked.head(10))
    print("Stacked shape:", df_stacked.shape)
    
    # Unstack index to columns
    df_unstacked = df_stacked.unstack()
    print("\nAfter unstacking:")
    print(df_unstacked)
    
    # Reset and set index
    print("\n--- Index Manipulation ---")
    df_reset = df.reset_index()
    print("After reset_index:")
    print(df_reset)
    
    df_set = df_reset.set_index(['first', 'second'])
    print("\nAfter set_index:")
    print(df_set)
    
    # Index operations
    print("\n--- Index Operations ---")
    print("Index levels:", df.index.levels)
    print("Index names:", df.index.names)
    
    # Swap levels
    df_swapped = df.swaplevel('first', 'second')
    print("After swapping levels:")
    print(df_swapped)
    
    return df

# Uncomment to run:
# df_multi = multiindex_operations()

# =============================================================================
# SECTION 14: WINDOW FUNCTIONS AND ROLLING OPERATIONS
# =============================================================================

def window_operations():
    print("\n=== WINDOW FUNCTIONS AND ROLLING OPERATIONS ===")
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    ts = pd.Series(np.random.randn(100).cumsum() + 100, index=dates, name='value')
    
    print("Original time series (first 10 values):")
    print(ts.head(10))
    
    # Rolling window operations
    print("\n--- Rolling Window Operations ---")
    
    # Simple rolling mean
    rolling_mean = ts.rolling(window=7).mean()
    print("7-day rolling mean (first 10 values):")
    print(rolling_mean.head(10))
    
    # Multiple rolling statistics
    rolling_stats = pd.DataFrame({
        'original': ts,
        'rolling_mean_7': ts.rolling(7).mean(),
        'rolling_std_7': ts.rolling(7).std(),
        'rolling_min_7': ts.rolling(7).min(),
        'rolling_max_7': ts.rolling(7).max()
    })
    
    print("\nRolling statistics (last 10 values):")
    print(rolling_stats.tail(10))
    
    # Expanding window operations
    print("\n--- Expanding Window Operations ---")
    expanding_stats = pd.DataFrame({
        'original': ts,
        'expanding_mean': ts.expanding().mean(),
        'expanding_std': ts.expanding().std(),
        'expanding_min': ts.expanding().min(),
        'expanding_max': ts.expanding().max()
    })
    
    print("Expanding statistics (last 10 values):")
    print(expanding_stats.tail(10))
    
    # Custom rolling functions
    print("\n--- Custom Rolling Functions ---")
    
    def rolling_percentile_90(x):
        return np.percentile(x, 90)
    
    custom_rolling = pd.DataFrame({
        'original': ts,
        'rolling_90th_percentile': ts.rolling(10).apply(rolling_percentile_90),
        'rolling_range': ts.rolling(10).apply(lambda x: x.max() - x.min()),
        'rolling_skew': ts.rolling(10).skew()
    })
    
    print("Custom rolling functions (last 10 values):")
    print(custom_rolling.tail(10))
    
    # EWM (Exponentially Weighted Moving) operations
    print("\n--- Exponentially Weighted Moving Operations ---")
    ewm_stats = pd.DataFrame({
        'original': ts,
        'ewm_mean': ts.ewm(span=10).mean(),
        'ewm_std': ts.ewm(span=10).std(),
        'ewm_var': ts.ewm(span=10).var()
    })
    
    print("EWM statistics (last 10 values):")
    print(ewm_stats.tail(10))
    
    # Plotting rolling statistics
    print("\n--- Plotting Rolling Statistics ---")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(ts.index, ts, label='Original', alpha=0.7)
    plt.plot(rolling_mean.index, rolling_mean, label='7-day Rolling Mean', linewidth=2)
    plt.plot(ewm_stats.index, ewm_stats['ewm_mean'], label='EWM Mean', linewidth=2)
    plt.title('Time Series with Moving Averages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(rolling_stats.index, rolling_stats['rolling_std_7'], label='7-day Rolling Std')
    plt.plot(ewm_stats.index, ewm_stats['ewm_std'], label='EWM Std')
    plt.title('Rolling Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ts, rolling_stats, ewm_stats

# Uncomment to run:
# ts_data, rolling_data, ewm_data = window_operations()

# =============================================================================
# SECTION 15: CATEGORICAL DATA AND MEMORY OPTIMIZATION
# =============================================================================

def categorical_operations():
    print("\n=== CATEGORICAL DATA OPERATIONS ===")
    
    # Create sample data with repeated string values
    np.random.seed(42)
    n = 10000
    categories = ['Category_A', 'Category_B', 'Category_C', 'Category_D', 'Category_E']
    df = pd.DataFrame({
        'category': np.random.choice(categories, n),
        'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3'], n),
        'value': np.random.randn(n)
    })
    
    print("Original DataFrame info:")
    print(df.info(memory_usage='deep'))
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Convert to categorical
    print("\n--- Converting to Categorical ---")
    df_cat = df.copy()
    df_cat['category'] = df_cat['category'].astype('category')
    df_cat['subcategory'] = df_cat['subcategory'].astype('category')
    
    print("After converting to categorical:")
    print(df_cat.info(memory_usage='deep'))
    print(f"Memory usage: {df_cat.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    memory_savings = (df.memory_usage(deep=True).sum() - df_cat.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum() * 100
    print(f"Memory savings: {memory_savings:.1f}%")
    
    # Categorical operations
    print("\n--- Categorical Operations ---")
    print("Categories in 'category' column:")
    print(df_cat['category'].cat.categories)
    
    print("\nValue counts:")
    print(df_cat['category'].value_counts())
    
    # Add new category
    df_cat['category'] = df_cat['category'].cat.add_categories(['Category_F'])
    print("\nAfter adding Category_F:")
    print(df_cat['category'].cat.categories)
    
    # Remove unused categories
    df_cat['category'] = df_cat['category'].cat.remove_unused_categories()
    print("\nAfter removing unused categories:")
    print(df_cat['category'].cat.categories)
    
    # Reorder categories
    new_order = ['Category_C', 'Category_A', 'Category_B', 'Category_D', 'Category_E']
    df_cat['category'] = df_cat['category'].cat.reorder_categories(new_order)
    print("\nAfter reordering:")
    print(df_cat['category'].cat.categories)
    
    # Ordered categorical
    print("\n--- Ordered Categorical ---")
    sizes = ['Small', 'Medium', 'Large', 'Extra Large']
    size_data = np.random.choice(sizes, 100)
    
    # Unordered categorical
    unordered_cat = pd.Categorical(size_data)
    print("Unordered categorical:")
    print(f"Can compare: {(unordered_cat[0] < unordered_cat[1]) if len(unordered_cat) > 1 else 'N/A'}")
    
    # Ordered categorical
    ordered_cat = pd.Categorical(size_data, categories=sizes, ordered=True)
    print("\nOrdered categorical:")
    print(f"Categories: {ordered_cat.categories}")
    print(f"Can compare: {ordered_cat[0] < ordered_cat[1] if len(ordered_cat) > 1 else 'N/A'}")
    
    # Groupby with categorical
    print("\n--- GroupBy with Categorical ---")
    grouped = df_cat.groupby('category')['value'].agg(['mean', 'count'])
    print("Grouped statistics:")
    print(grouped)
    
    return df_cat

# Uncomment to run:
# df_categorical = categorical_operations()

# =============================================================================
# SECTION 16: ADVANCED MATPLOTLIB CUSTOMIZATION
# =============================================================================

def advanced_plotting():
    print("\n=== ADVANCED MATPLOTLIB CUSTOMIZATION ===")
    
    # Set up custom style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Create sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
    y3 = np.sin(x + np.pi/4) + np.random.normal(0, 0.1, 100)
    
    # Advanced subplot layouts
    print("--- Advanced Subplot Layouts ---")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create complex subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main plot (spans multiple cells)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_main.plot(x, y1, label='sin(x)', linewidth=2, alpha=0.8)
    ax_main.plot(x, y2, label='cos(x)', linewidth=2, alpha=0.8)
    ax_main.fill_between(x, y1, y2, alpha=0.3)
    ax_main.set_title('Main Plot with Fill Between')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Side histogram
    ax_hist = fig.add_subplot(gs[0:2, 2])
    ax_hist.hist(y1, bins=20, alpha=0.7, orientation='horizontal', color='blue')
    ax_hist.hist(y2, bins=20, alpha=0.7, orientation='horizontal', color='orange')
    ax_hist.set_title('Distributions')
    ax_hist.set_xlabel('Frequency')
    
    # Bottom left - scatter plot
    ax_scatter = fig.add_subplot(gs[2, 0])
    colors = np.random.rand(100)
    ax_scatter.scatter(y1, y2, c=colors, alpha=0.6, cmap='viridis')
    ax_scatter.set_title('Scatter Plot')
    ax_scatter.set_xlabel('sin(x)')
    ax_scatter.set_ylabel('cos(x)')
    
    # Bottom middle - box plot
    ax_box = fig.add_subplot(gs[2, 1])
    ax_box.boxplot([y1, y2, y3], labels=['sin(x)', 'cos(x)', 'sin(x+π/4)'])
    ax_box.set_title('Box Plots')
    ax_box.tick_params(axis='x', rotation=45)
    
    # Bottom right - polar plot
    ax_polar = fig.add_subplot(gs[2, 2], projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3 * np.sin(5*theta)
    ax_polar.plot(theta, r)
    ax_polar.fill(theta, r, alpha=0.3)
    ax_polar.set_title('Polar Plot')
    
    plt.suptitle('Advanced Subplot Layout', fontsize=16, fontweight='bold')
    plt.show()
    
    # Custom annotations and text
    print("\n--- Custom Annotations ---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    line1, = ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    line2, = ax.plot(x, y2, 'r-', linewidth=2, label='cos(x)')
    
    # Find intersection points (approximately)
    intersections = []
    for i in range(len(x)-1):
        if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) < 0:
            intersections.append((x[i], y1[i]))
    
    # Annotate intersection points
    for i, (xi, yi) in enumerate(intersections[:3]):  # First 3 intersections
        ax.annotate(f'Intersection {i+1}', 
                   xy=(xi, yi), xytext=(xi+0.5, yi+0.5),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=10)
        ax.plot(xi, yi, 'ko', markersize=8)
    
    # Add text box
    textstr = 'Key Statistics:\n' + f'sin(x) mean: {np.mean(y1):.3f}\n' + f'cos(x) mean: {np.mean(y2):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_title('Plot with Custom Annotations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    # 3D plotting
    print("\n--- 3D Plotting ---")
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    X = np.linspace(-2, 2, 50)
    Y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('3D Surface Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 3D scatter plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    n = 500
    x_3d = np.random.randn(n)
    y_3d = np.random.randn(n)
    z_3d = x_3d**2 + y_3d**2 + np.random.randn(n) * 0.1
    colors = z_3d
    
    scatter = ax2.scatter(x_3d, y_3d, z_3d, c=colors, cmap='plasma', alpha=0.6)
    ax2.set_title('3D Scatter Plot')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.colorbar(scatter, ax=ax2, shrink=0.5)
    plt.tight_layout()
    plt.show()

# Uncomment to run:
# advanced_plotting()

# =============================================================================
# SECTION 17: FINAL INTEGRATION EXAMPLE
# =============================================================================

def comprehensive_analysis():
    print("\n=== COMPREHENSIVE DATA ANALYSIS EXAMPLE ===")
    
    # Create realistic dataset
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    # Generate realistic business data
    base_sales = 1000
    seasonal_effect = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    weekly_effect = 100 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = np.random.normal(0, 50, n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': base_sales + seasonal_effect + weekly_effect + noise,
        'marketing_spend': np.random.normal(100, 20, n_days),
        'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 3, n_days),
        'day_of_week': dates.day_name(),
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': dates.weekday >= 5
    })
    
    # Add some business logic
    df['marketing_efficiency'] = df['sales'] / df['marketing_spend']
    df['sales_category'] = pd.cut(df['sales'], bins=3, labels=['Low', 'Medium', 'High'])
    
    print("Dataset overview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Comprehensive analysis
    print("\n--- Time Series Analysis ---")
    
    # Set date as index
    df_ts = df.set_index('date')
    
    # Monthly aggregation
    monthly_stats = df_ts.resample('M').agg({
        'sales': ['mean', 'sum', 'std'],
        'marketing_spend': 'mean',
        'temperature': 'mean',
        'marketing_efficiency': 'mean'
    })
    
    print("Monthly statistics (first 6 months):")
    print(monthly_stats.head(6))
    
    # Day of week analysis
    dow_analysis = df.groupby('day_of_week').agg({
        'sales': ['mean', 'std'],
        'marketing_efficiency': 'mean'
    }).round(2)
    
    # Reorder by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis = dow_analysis.reindex(day_order)
    
    print("\nDay of week analysis:")
    print(dow_analysis)
    
    # Correlation analysis
    correlation_matrix = df[['sales', 'marketing_spend', 'temperature']].corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Time series plot
    axes[0, 0].plot(df['date'], df['sales'], alpha=0.7, linewidth=1)
    axes[0, 0].plot(df['date'], df['sales'].rolling(30).mean(), 'r-', linewidth=2, label='30-day MA')
    axes[0, 0].set_title('Daily Sales with Moving Average')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Monthly sales
    monthly_sales = df.groupby('month')['sales'].mean()
    axes[0, 1].bar(monthly_sales.index, monthly_sales.values)
    axes[0, 1].set_title('Average Sales by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Sales')
    
    # Day of week analysis
    dow_sales = df.groupby('day_of_week')['sales'].mean().reindex(day_order)
    axes[0, 2].bar(range(len(dow_sales)), dow_sales.values)
    axes[0, 2].set_title('Average Sales by Day of Week')
    axes[0, 2].set_xlabel('Day of Week')
    axes[0, 2].set_ylabel('Average Sales')
    axes[0, 2].set_xticks(range(len(day_order)))
    axes[0, 2].set_xticklabels([day[:3] for day in day_order], rotation=45)
    
    # Scatter plot: sales vs marketing spend
    axes[1, 0].scatter(df['marketing_spend'], df['sales'], alpha=0.6, c=df['temperature'], cmap='coolwarm')
    axes[1, 0].set_title('Sales vs Marketing Spend (colored by temperature)')
    axes[1, 0].set_xlabel('Marketing Spend')
    axes[1, 0].set_ylabel('Sales')
    
    # Box plot: sales by category
    sales_by_category = [df[df['sales_category'] == cat]['sales'].values for cat in ['Low', 'Medium', 'High']]
    axes[1, 1].boxplot(sales_by_category, labels=['Low', 'Medium', 'High'])
    axes[1, 1].set_title('Sales Distribution by Category')
    axes[1, 1].set_xlabel('Sales Category')
    axes[1, 1].set_ylabel('Sales')
    
    # Correlation heatmap
    im = axes[1, 2].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Heatmap')
    axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
    axes[1, 2].set_yticks(range(len(correlation_matrix.columns)))
    axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
    axes[1, 2].set_yticklabels(correlation_matrix.columns)
    
    # Add correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', 
                           color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.suptitle('Comprehensive Business Data Analysis', fontsize=16, y=1.02)
    plt.show()
    
    print("\n--- Key Insights ---")
    insights = [
        f"1. Average daily sales: ${df['sales'].mean():.0f}",
        f"2. Best performing day: {dow_sales.idxmax()} (${dow_sales.max():.0f})",
        f"3. Worst performing day: {dow_sales.idxmin()} (${dow_sales.min():.0f})",
        f"4. Sales-Marketing correlation: {correlation_matrix.loc['sales', 'marketing_spend']:.3f}",
        f"5. Sales-Temperature correlation: {correlation_matrix.loc['sales', 'temperature']:.3f}",
        f"6. Weekend vs Weekday sales difference: ${df[df['is_weekend']]['sales'].mean() - df[~df['is_weekend']]['sales'].mean():.0f}",
        f"7. Highest sales month: {monthly_sales.idxmax()} (${monthly_sales.max():.0f})",
        f"8. Sales volatility (std): ${df['sales'].std():.0f}"
    ]
    
    for insight in insights:
        print(insight)
    
    return df

# Uncomment to run:
# df_comprehensive = comprehensive_analysis()

print("\n" + "="*80)
print("STUDY GUIDE COMPLETE!")
print("="*80)
print("This comprehensive guide covers:")
print("✓ Basic pandas operations and data selection")
print("✓ DateTime operations and time series analysis")
print("✓ Resampling and frequency conversion")
print("✓ GroupBy operations and aggregations")
print("✓ Data cleaning and manipulation")
print("✓ Merging, joining, and concatenation")
print("✓ Pivot tables and crosstabs")
print("✓ MultiIndex operations")
print("✓ Window functions and rolling operations")
print("✓ Categorical data optimization")
print("✓ Basic to advanced matplotlib plotting")
print("✓ Performance optimization tips")
print("✓ Comprehensive real-world analysis example")
print("\nUncomment any section to run and practice!")
print("="*80)