import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

# Note: seaborn is NOT imported here

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Sleep Chronotype Analyzer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chronotype-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        animation: slideIn 0.5s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    @keyframes slideIn {
        from {transform: translateX(-100px); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate sample data if no data exists
def generate_sample_data():
    """Generate sample sleep data"""
    np.random.seed(42)
    
    chronotypes = ['Lion', 'Bear', 'Wolf', 'Dolphin']
    data = []
    
    # Define patterns for each chronotype
    patterns = {
        'Lion': {'bedtime': 22, 'wakeup': 5.5, 'quality': 8},
        'Bear': {'bedtime': 23, 'wakeup': 7, 'quality': 7.5},
        'Wolf': {'bedtime': 1, 'wakeup': 8.5, 'quality': 7},
        'Dolphin': {'bedtime': 0, 'wakeup': 6.5, 'quality': 6}
    }
    
    # Generate 60 days of data
    for chrono in chronotypes:
        pattern = patterns[chrono]
        for day in range(60):
            date = datetime.now() - timedelta(days=60-day)
            
            # Add some variation
            bedtime = pattern['bedtime'] + np.random.normal(0, 0.8)
            wakeup = pattern['wakeup'] + np.random.normal(0, 0.8)
            
            # Keep in valid ranges
            bedtime = bedtime % 24
            wakeup = np.clip(wakeup, 4, 12)
            
            # Calculate sleep duration
            if wakeup > bedtime:
                sleep_duration = wakeup - bedtime
            else:
                sleep_duration = (24 - bedtime) + wakeup
            
            # Add quality variation
            quality = pattern['quality'] + np.random.normal(0, 1)
            quality = np.clip(quality, 1, 10)
            
            # Disturbances
            disturbances = np.random.poisson(0.5 if chrono != 'Dolphin' else 1)
            
            data.append({
                'date': date,
                'chronotype': chrono,
                'bedtime_hour': round(bedtime, 2),
                'wakeup_hour': round(wakeup, 2),
                'sleep_duration': round(sleep_duration, 2),
                'sleep_quality': round(quality, 2),
                'disturbances': disturbances,
                'user_id': chronotypes.index(chrono) + 1
            })
    
    df = pd.DataFrame(data)
    return df

# Initialize session state
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
    st.session_state.model_loaded = False

# Title
st.markdown('<h1 class="main-header">🌙 AI-Powered Sleep Chronotype Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover your chronotype using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Controls")
    
    # ML Option (disabled if no model)
    use_ml = st.checkbox("🤖 Use Machine Learning", value=False, 
                        help="Enable AI-powered predictions (train model first)")
    
    if use_ml:
        st.info("💡 ML model not trained yet. Using rule-based classification.")
        use_ml = False
    
    st.markdown("---")
    
    # Data generation
    st.markdown("### 📁 Data Management")
    if st.button("🔄 Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            df = generate_sample_data()
            # Save to session state
            st.session_state.df = df
            st.success(f"✅ Generated {len(df)} sample records!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About Chronotypes")
    st.info("""
    **Lion 🦁** - Early riser (5-6 AM), most productive in morning  
    **Bear 🐻** - Follows sun cycle (7-8 AM), standard work hours  
    **Wolf 🐺** - Night owl (9-10 AM), creative at night  
    **Dolphin 🐬** - Irregular sleeper, light sleeper
    """)

# Load or generate data
if 'df' not in st.session_state:
    # Try to load existing data
    try:
        if os.path.exists('data/sleep_data.csv'):
            df = pd.read_csv('data/sleep_data.csv')
            df['date'] = pd.to_datetime(df['date'])
            st.session_state.df = df
        else:
            # Generate sample data
            df = generate_sample_data()
            st.session_state.df = df
    except Exception as e:
        st.warning(f"Could not load data: {e}. Generating sample data...")
        df = generate_sample_data()
        st.session_state.df = df

df = st.session_state.df

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔍 Personal Analysis", "📈 Trends", "💡 Recommendations"
])

# Tab 1: Dashboard
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Avg Sleep Duration", f"{df['sleep_duration'].mean():.1f} hrs")
    with col3:
        st.metric("Avg Sleep Quality", f"{df['sleep_quality'].mean():.1f}/10")
    with col4:
        st.metric("Avg Bedtime", f"{df['bedtime_hour'].mean():.1f}:00")
    
    # Chronotype distribution
    st.subheader("📊 Chronotype Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        chrono_counts = df['chronotype'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#FFD700', '#8B7355', '#708090', '#4A90E2']
        wedges, texts, autotexts = ax.pie(chrono_counts.values, 
                                           labels=chrono_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors[:len(chrono_counts)],
                                           explode=[0.05]*len(chrono_counts))
        ax.set_title('Sleep Chronotypes in Population')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Sleep Metrics by Chronotype")
        metrics = df.groupby('chronotype').agg({
            'sleep_duration': 'mean',
            'sleep_quality': 'mean',
            'bedtime_hour': 'mean',
            'wakeup_hour': 'mean'
        }).round(2)
        st.dataframe(metrics, use_container_width=True)
    
    # Sleep quality by chronotype (using matplotlib instead of seaborn)
    st.subheader("🎯 Sleep Quality Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot manually using matplotlib
    chronotypes = df['chronotype'].unique()
    box_data = [df[df['chronotype'] == ct]['sleep_quality'].values for ct in chronotypes]
    
    bp = ax.boxplot(box_data, labels=chronotypes, patch_artist=True)
    
    # Color the boxes
    colors = ['#FFD700', '#8B7355', '#708090', '#4A90E2']
    for patch, color in zip(bp['boxes'], colors[:len(chronotypes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Sleep Quality Distribution by Chronotype')
    ax.set_xlabel('Chronotype')
    ax.set_ylabel('Sleep Quality (1-10)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Tab 2: Personal Analysis
with tab2:
    st.subheader("🔍 Find Your Chronotype")
    
    # Rule-based classification function
    def classify_chronotype(bedtime, wakeup):
        if bedtime < 23 and wakeup < 7:
            return "Lion 🦁"
        elif bedtime < 24 and wakeup < 9:
            return "Bear 🐻"
        elif bedtime >= 24 and wakeup > 8:
            return "Wolf 🐺"
        else:
            return "Dolphin 🐬"
    
    col1, col2 = st.columns(2)
    
    with col1:
        bedtime = st.time_input("⏰ Bedtime", datetime.now().replace(hour=23, minute=0))
        bedtime_hour = bedtime.hour + bedtime.minute/60
        
        sleep_quality = st.slider("💤 Sleep Quality (1-10)", 1, 10, 7)
        disturbances = st.number_input("😴 Night Disturbances", 0, 10, 1)
        
    with col2:
        wakeup = st.time_input("🌅 Wake-up Time", datetime.now().replace(hour=7, minute=0))
        wakeup_hour = wakeup.hour + wakeup.minute/60
        
        st.markdown("---")
        st.markdown("### 📊 Your Input")
        st.write(f"**Bedtime:** {bedtime_hour:.1f} hours")
        st.write(f"**Wake-up:** {wakeup_hour:.1f} hours")
        
        sleep_duration = (wakeup_hour - bedtime_hour) if wakeup_hour > bedtime_hour else (24 - bedtime_hour + wakeup_hour)
        st.write(f"**Sleep Duration:** {sleep_duration:.1f} hours")
    
    if st.button("🎯 Analyze My Sleep Pattern", type="primary"):
        chronotype = classify_chronotype(bedtime_hour, wakeup_hour)
        
        st.markdown(f"""
        <div class="chronotype-card">
            <h2>🎉 Your Chronotype: {chronotype}</h2>
            <p>Based on your sleep pattern analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show similar users
        st.subheader("👥 How you compare to others")
        similar_users = df[df['chronotype'] == chronotype.split()[0]]
        if len(similar_users) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Sleep Duration", f"{similar_users['sleep_duration'].mean():.1f} hrs")
            with col2:
                st.metric("Avg Sleep Quality", f"{similar_users['sleep_quality'].mean():.1f}/10")
            with col3:
                st.metric("Avg Bedtime", f"{similar_users['bedtime_hour'].mean():.1f}:00")

# Tab 3: Trends
with tab3:
    st.subheader("📈 Sleep Pattern Trends")
    
    # Prepare data for plotting
    plot_df = df.copy()
    plot_df['date_str'] = plot_df['date'].dt.strftime('%Y-%m-%d')
    
    # Line plot using matplotlib
    st.subheader("Sleep Duration Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for chrono in plot_df['chronotype'].unique():
        chrono_data = plot_df[plot_df['chronotype'] == chrono]
        ax.plot(chrono_data['date'], chrono_data['sleep_duration'], 
                label=chrono, marker='o', markersize=3, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sleep Duration (hours)')
    ax.set_title('Sleep Duration Trends Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Heatmap using matplotlib instead of seaborn
    st.subheader("🗓️ Sleep Quality Heatmap")
    
    # Create day of week
    plot_df['day_of_week'] = plot_df['date'].dt.day_name()
    
    # Create pivot table for heatmap
    pivot_data = plot_df.pivot_table(values='sleep_quality', 
                                    index='day_of_week', 
                                    columns='chronotype',
                                    aggfunc='mean')
    
    # Create custom heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Sleep Quality')
    
    # Set labels
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.values[i, j]:.1f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Average Sleep Quality by Chronotype and Day')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribution plots
    st.subheader("📊 Sleep Metrics Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for chrono in plot_df['chronotype'].unique():
            data = plot_df[plot_df['chronotype'] == chrono]['sleep_duration']
            ax.hist(data, alpha=0.5, label=chrono, bins=15)
        ax.set_xlabel('Sleep Duration (hours)')
        ax.set_ylabel('Frequency')
        ax.set_title('Sleep Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for chrono in plot_df['chronotype'].unique():
            data = plot_df[plot_df['chronotype'] == chrono]['sleep_quality']
            ax.hist(data, alpha=0.5, label=chrono, bins=15)
        ax.set_xlabel('Sleep Quality')
        ax.set_ylabel('Frequency')
        ax.set_title('Sleep Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 4: Recommendations
with tab4:
    st.subheader("💡 Personalized Sleep Recommendations")
    
    # General tips
    st.info("""
    ### 🌟 Universal Sleep Hygiene Tips
    - Maintain consistent sleep schedule even on weekends
    - Avoid screens 1 hour before bedtime
    - Keep bedroom cool (65-68°F / 18-20°C)
    - Avoid caffeine after 2 PM
    - Get morning sunlight exposure
    - Exercise regularly but not too close to bedtime
    """)
    
    # Chronotype-specific recommendations
    st.subheader("📌 Chronotype-Specific Advice")
    
    rec_df = pd.DataFrame({
        'Chronotype': ['Lion 🦁', 'Bear 🐻', 'Wolf 🐺', 'Dolphin 🐬'],
        'Optimal Schedule': ['5 AM - 2 PM', '8 AM - 5 PM', '12 PM - 9 PM', '10 AM - 6 PM'],
        'Best For': ['Morning tasks', 'Collaborative work', 'Creative work', 'Flexible tasks'],
        'Exercise Time': ['6-7 AM', '10-11 AM', '6-8 PM', '12-2 PM'],
        'Avoid': ['Late nights', 'Early meetings', 'Morning calls', 'Rigid schedules']
    })
    
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
    
    # Productivity tips
    st.subheader("🚀 Productivity Tips by Chronotype")
    
    with st.expander("🦁 Lion Chronotype Tips"):
        st.markdown("""
        - **Peak hours**: 6 AM - 12 PM
        - **Best for**: Strategic planning, important decisions
        - **Break times**: 10:30 AM and 2:30 PM
        - **Avoid**: Late-night work sessions
        - **Ideal meeting times**: 8 AM - 11 AM
        """)
    
    with st.expander("🐻 Bear Chronotype Tips"):
        st.markdown("""
        - **Peak hours**: 10 AM - 4 PM
        - **Best for**: Team collaboration, meetings
        - **Break times**: 11 AM and 3 PM
        - **Avoid**: Very early morning tasks
        - **Ideal meeting times**: 10 AM - 12 PM
        """)
    
    with st.expander("🐺 Wolf Chronotype Tips"):
        st.markdown("""
        - **Peak hours**: 4 PM - 10 PM
        - **Best for**: Deep work, creative tasks
        - **Break times**: 6 PM and 9 PM
        - **Avoid**: Morning meetings before 10 AM
        - **Ideal meeting times**: 1 PM - 3 PM
        """)
    
    with st.expander("🐬 Dolphin Chronotype Tips"):
        st.markdown("""
        - **Peak hours**: 11 AM - 5 PM
        - **Best for**: Problem-solving, creative thinking
        - **Break times**: Take frequent short breaks
        - **Avoid**: Rigid schedules
        - **Ideal meeting times**: 11 AM - 1 PM
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌙 Sleep Chronotype Analyzer | Based on Sleep Pattern Analysis | Made with Streamlit</p>
    <p>💡 <strong>Tip:</strong> Generate sample data to see the analysis in action!</p>
</div>
""", unsafe_allow_html=True)