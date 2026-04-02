import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sleep_data(n_days=30, n_users=4):
    """Generate synthetic sleep data for multiple users"""
    
    chronotypes = {
        'Lion': {'bedtime': 22.0, 'wakeup': 5.5, 'variation': 0.5, 'quality_mean': 8.0},
        'Bear': {'bedtime': 23.0, 'wakeup': 7.0, 'variation': 1.0, 'quality_mean': 7.5},
        'Wolf': {'bedtime': 1.0, 'wakeup': 8.5, 'variation': 1.5, 'quality_mean': 7.0},
        'Dolphin': {'bedtime': 0.0, 'wakeup': 6.5, 'variation': 1.2, 'quality_mean': 6.0}
    }
    
    data = []
    
    for user_id, (chronotype, pattern) in enumerate(chronotypes.items(), 1):
        for day in range(n_days):
            # Add some random variation
            bedtime = pattern['bedtime'] + np.random.normal(0, pattern['variation'])
            wakeup = pattern['wakeup'] + np.random.normal(0, pattern['variation'])
            
            # Ensure bedtime is between 20-28 (8 PM - 4 AM)
            bedtime = np.clip(bedtime, 20, 28) % 24
            wakeup = np.clip(wakeup, 4, 12)
            
            # Calculate sleep duration
            sleep_duration = (wakeup - bedtime) if wakeup > bedtime else (24 - bedtime + wakeup)
            
            # Add sleep quality score (1-10)
            sleep_quality = np.random.normal(pattern['quality_mean'], 1.5)
            sleep_quality = np.clip(sleep_quality, 1, 10)
            
            # Add some random disturbances
            disturbances = np.random.poisson(1 if chronotype == 'Dolphin' else 0.5)
            
            # Add REM sleep percentage
            rem_percentage = np.random.normal(22, 3) if sleep_quality > 7 else np.random.normal(18, 4)
            rem_percentage = np.clip(rem_percentage, 10, 35)
            
            date = datetime.now() - timedelta(days=n_days-day)
            
            # Add day of week effect
            day_of_week = date.strftime('%A')
            weekend_effect = 1.2 if day_of_week in ['Saturday', 'Sunday'] else 1.0
            bedtime = bedtime * weekend_effect if weekend_effect > 1 else bedtime
            
            data.append({
                'user_id': user_id,
                'chronotype': chronotype,
                'date': date,
                'bedtime_hour': round(bedtime, 2),
                'wakeup_hour': round(wakeup, 2),
                'sleep_duration': round(sleep_duration, 2),
                'sleep_quality': round(sleep_quality, 2),
                'disturbances': disturbances,
                'rem_percentage': round(rem_percentage, 2),
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week in ['Saturday', 'Sunday'] else 0
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_sleep_data(100)
    df.to_csv('data/sleep_data.csv', index=False)
    print(f"✅ Generated {len(df)} sleep records")
    print(f"📊 Chronotype distribution:\n{df['chronotype'].value_counts()}")