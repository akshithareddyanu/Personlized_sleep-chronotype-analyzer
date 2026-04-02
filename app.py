import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sleep Chronotype Analyzer",
    page_icon="🌙",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chronotype-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate sample data
def generate_sample_data():
    """Generate sample sleep data"""
    np.random.seed(42)
    
    chronotypes = ['Lion', 'Bear', 'Wolf', 'Dolphin']
    data = []
    
    patterns = {
        'Lion': {'bedtime': 22, 'wakeup': 5.5, 'quality': 8},
        'Bear': {'bedtime': 23, 'wakeup': 7, 'quality': 7.5},
        'Wolf': {'bedtime': 1, 'wakeup': 8.5, 'quality': 7},
        'Dolphin': {'bedtime': 0, 'wakeup': 6.5, 'quality': 6}
    }
    
    for chrono in chronotypes:
        pattern = patterns[chrono]
        for day in range(60):
            date = datetime.now() - timedelta(days=60-day)
            
            bedtime = pattern['bedtime'] + np.random.normal(0, 0.8)
            wakeup = pattern['wakeup'] + np.random.normal(0, 0.8)
            bedtime = bedtime % 24
            wakeup = np.clip(wakeup, 4, 12)
            
            if wakeup > bedtime:
                sleep_duration = wakeup - bedtime
            else:
                sleep_duration = (24 - bedtime) + wakeup
            
            quality = np.clip(pattern['quality'] + np.random.normal(0, 1), 1, 10)
            disturbances = np.random.poisson(0.5 if chrono != 'Dolphin' else 1)
            
            data.append({
                'date': date,
                'chronotype': chrono,
                'bedtime_hour': round(bedtime, 2),
                'wakeup_hour': round(wakeup, 2),
                'sleep_duration': round(sleep_duration, 2),
                'sleep_quality': round(quality, 2),
                'disturbances': disturbances
            })
    
    return pd.DataFrame(data)

# Title
st.markdown('<h1 class="main-header">🌙 Sleep Chronotype Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Controls")
    
    if st.button("🔄 Generate Sample Data"):
        with st.spinner("Generating data..."):
            df = generate_sample_data()
            st.session_state.df = df
            st.success(f"✅ Generated {len(df)} records!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About Chronotypes")
    st.info("""
    **Lion 🦁** - Early riser, productive in morning  
    **Bear 🐻** - Follows sun cycle, standard hours  
    **Wolf 🐺** - Night owl, creative at night  
    **Dolphin 🐬** - Irregular/light sleeper
    """)

# Initialize or load data
if 'df' not in st.session_state:
    df = generate_sample_data()
    st.session_state.df = df

df = st.session_state.df
df['date'] = pd.to_datetime(df['date'])

# Dashboard metrics
st.subheader("📊 Sleep Statistics Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Avg Sleep Duration", f"{df['sleep_duration'].mean():.1f} hrs")
with col3:
    st.metric("Avg Sleep Quality", f"{df['sleep_quality'].mean():.1f}/10")
with col4:
    st.metric("Avg Bedtime", f"{df['bedtime_hour'].mean():.1f}:00")

# Two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Chronotype Distribution")
    chrono_counts = df['chronotype'].value_counts().reset_index()
    chrono_counts.columns = ['Chronotype', 'Count']
    fig = px.pie(chrono_counts, values='Count', names='Chronotype', 
                 title='Distribution of Chronotypes',
                 color_discrete_sequence=['#FFD700', '#8B7355', '#708090', '#4A90E2'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Sleep Metrics by Chronotype")
    metrics = df.groupby('chronotype').agg({
        'sleep_duration': 'mean',
        'sleep_quality': 'mean',
        'bedtime_hour': 'mean'
    }).round(2)
    metrics.columns = ['Sleep Duration', 'Sleep Quality', 'Bedtime']
    st.dataframe(metrics, use_container_width=True)

# Trends
st.subheader("📈 Sleep Duration Trends")
fig = px.line(df, x='date', y='sleep_duration', color='chronotype',
              title='Sleep Duration Over Time',
              labels={'sleep_duration': 'Hours', 'date': 'Date'})
st.plotly_chart(fig, use_container_width=True)

# Sleep quality distribution
st.subheader("🎯 Sleep Quality Distribution by Chronotype")
fig = px.box(df, x='chronotype', y='sleep_quality', color='chronotype',
             title='Sleep Quality Distribution',
             labels={'sleep_quality': 'Sleep Quality (1-10)', 'chronotype': 'Chronotype'})
st.plotly_chart(fig, use_container_width=True)

# Personal analysis
st.subheader("🔍 Find Your Chronotype")

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

with col2:
    wakeup = st.time_input("🌅 Wake-up Time", datetime.now().replace(hour=7, minute=0))
    wakeup_hour = wakeup.hour + wakeup.minute/60

if st.button("🎯 Analyze My Pattern", type="primary"):
    chronotype = classify_chronotype(bedtime_hour, wakeup_hour)
    
    st.markdown(f"""
    <div class="chronotype-card">
        <h2>🎉 Your Chronotype: {chronotype}</h2>
        <p>Based on your bedtime and wake-up time analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show comparison
    st.subheader("📊 How You Compare")
    similar = df[df['chronotype'] == chronotype.split()[0]]
    if len(similar) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sleep Duration", f"{similar['sleep_duration'].mean():.1f} hrs")
        with col2:
            st.metric("Avg Sleep Quality", f"{similar['sleep_quality'].mean():.1f}/10")
        with col3:
            st.metric("Avg Bedtime", f"{similar['bedtime_hour'].mean():.1f}:00")

# Recommendations
st.subheader("💡 Sleep Recommendations")

with st.expander("🌟 Universal Sleep Tips"):
    st.markdown("""
    - Maintain consistent sleep schedule
    - Avoid screens 1 hour before bed
    - Keep bedroom cool (65-68°F)
    - Avoid caffeine after 2 PM
    - Get morning sunlight
    """)

with st.expander("📌 Chronotype-Specific Advice"):
    rec_data = {
        'Chronotype': ['Lion', 'Bear', 'Wolf', 'Dolphin'],
        'Best Schedule': ['5 AM - 2 PM', '8 AM - 5 PM', '12 PM - 9 PM', '10 AM - 6 PM'],
        'Peak Hours': ['6-11 AM', '9 AM-3 PM', '4-10 PM', '11 AM-5 PM'],
        'Exercise Time': ['6-7 AM', '10-11 AM', '6-8 PM', '12-2 PM']
    }
    st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🌙 Sleep Chronotype Analyzer | Made with Streamlit</p>", unsafe_allow_html=True)