import pandas as pd


def get_stats_target(df, group_column, target_column):
    df_old = df.copy().reset_index()
    grouped = df_old.groupby(group_column)
    the_stats = grouped[target_column].agg(['mean', 'median', 'max', 'min', 'std']).reset_index()

    the_stats.columns = [group_column,
                         '%s_mean_by_%s' % (target_column, group_column),
                         '%s_median_by_%s' % (target_column, group_column),
                         '%s_max_by_%s' % (target_column, group_column),
                         '%s_min_by_%s' % (target_column, group_column),
                         '%s_std_by_%s' % (target_column, group_column)]

    df_old = pd.merge(left=df_old, right=the_stats, on=group_column, how='left').set_index('id')
    return df_old


def create_new_features(all_df):
    all_df['floor_by_max_floor'] = all_df['floor'] / all_df['max_floor']
    all_df["extra_sq"] = all_df["full_sq"] - all_df["life_sq"]
    all_df['life_sq-kitchen_sq'] = all_df['life_sq'] - all_df['kitch_sq']

    # Room
    all_df['avg_room_size'] = (all_df['life_sq'] - all_df['kitch_sq']) / all_df['num_room']
    all_df['life_sq_prop'] = all_df['life_sq'] / all_df['full_sq']
    all_df['kitch_sq_prop'] = all_df['kitch_sq'] / all_df['full_sq']

    # Calculate age of building
    all_df['build_age'] = all_df['timestamp_year'] - all_df['build_year']
    all_df = all_df.drop('build_year', axis=1)

    # Population
    all_df['population_den'] = all_df['raion_popul'] / all_df['area_m']
    all_df['gender_rate'] = all_df['male_f'] / all_df['female_f']

    # Young
    all_df['young_rate'] = all_df['young_all'] / all_df['raion_popul']
    all_df['young_gender_rate'] = all_df['young_male'] / all_df['young_female']

    # Work
    all_df['working_rate'] = all_df['work_all'] / all_df['full_all']
    all_df['working_gender_rate'] = all_df['work_male'] / all_df['work_female']
    all_df['working_young_rate'] = all_df['young_all'] / all_df['work_all']

    # Old
    all_df['ekder_rate'] = all_df['ekder_all'] / all_df['raion_popul']
    all_df['ekder_gender_rate'] = all_df['ekder_male'] / all_df['ekder_female']

    # Education
    all_df['preschool_ratio'] = all_df['children_preschool'] / all_df['preschool_quota']
    all_df['school_ratio'] = all_df['children_school'] / all_df['school_quota']

    # NaNs count
    all_df['nan_count'] = all_df[['full_sq', 'build_age', 'life_sq', 'floor', 'max_floor', 'num_room']].isnull().sum(axis=1)

    # Statistical features
    all_df = get_stats_target(all_df, 'sub_area', 'max_floor')
    all_df = get_stats_target(all_df, 'sub_area', 'num_room')
    all_df = get_stats_target(all_df, 'sub_area', 'full_sq')
    all_df = get_stats_target(all_df, 'sub_area', 'life_sq')
    all_df = get_stats_target(all_df, 'sub_area', 'kitch_sq')

    # District features
    all_df['hospital_bed_density'] = all_df['hospital_beds_raion'] / all_df['raion_popul']
    all_df['healthcare_centers_density'] = all_df['healthcare_centers_raion'] / all_df['raion_popul']
    all_df['shopping_centers_density'] = all_df['shopping_centers_raion'] / all_df['raion_popul']
    all_df['university_top_20_density'] = all_df['university_top_20_raion'] / all_df['raion_popul']
    all_df['sport_objects_density'] = all_df['sport_objects_raion'] / all_df['raion_popul']
    all_df['best_university_ratio'] = all_df['university_top_20_raion'] / (all_df['sport_objects_raion'] + 1)
    all_df['good_bad_propotion'] = (all_df['sport_objects_raion'] + 1) / (all_df['additional_education_raion'] + 1)
    all_df['num_schools'] = all_df['sport_objects_raion'] + all_df['additional_education_raion']
    all_df['schools_density'] = all_df['num_schools'] + all_df['raion_popul']
    all_df['additional_education_density'] = all_df['additional_education_raion'] / all_df['raion_popul']

    all_df['raion_top_20_school'] = all_df['school_education_centers_top_20_raion'] / \
        all_df['school_education_centers_raion']

    all_df['congestion_metro'] = all_df['metro_km_avto'] / all_df['metro_min_avto']
    all_df['congestion_metro'].fillna(all_df['congestion_metro'].mean(), inplace=True)
    all_df['congestion_railroad'] = all_df['railroad_station_avto_km'] / all_df['railroad_station_avto_min']

    all_df['square_per_office_500'] = all_df['office_sqm_500'] / all_df['office_count_500']
    all_df['square_per_trc_500'] = all_df['trc_sqm_500'] / all_df['trc_count_500']
    all_df['square_per_office_1000'] = all_df['office_sqm_1000'] / all_df['office_count_1000']
    all_df['square_per_trc_1000'] = all_df['trc_sqm_1000'] / all_df['trc_count_1000']
    all_df['square_per_office_1500'] = all_df['office_sqm_1500'] / all_df['office_count_1500']
    all_df['square_per_trc_1500'] = all_df['trc_sqm_1500'] / all_df['trc_count_1500']
    all_df['square_per_office_2000'] = all_df['office_sqm_2000'] / all_df['office_count_2000']
    all_df['square_per_trc_2000'] = all_df['trc_sqm_2000'] / all_df['trc_count_2000']
    all_df['square_per_office_3000'] = all_df['office_sqm_3000'] / all_df['office_count_3000']
    all_df['square_per_trc_3000'] = all_df['trc_sqm_3000'] / all_df['trc_count_3000']
    all_df['square_per_office_5000'] = all_df['office_sqm_5000'] / all_df['office_count_5000']
    all_df['square_per_trc_5000'] = all_df['trc_sqm_5000'] / all_df['trc_count_5000']

    all_df['cafe_sum_500_diff'] = all_df['cafe_sum_500_max_price_avg'] - all_df['cafe_sum_500_min_price_avg']

    # The activity in the real estate market for each particular month is an important factor.
    # Thus, creating columns for month years and the no. of houses for each month year(month_year_cnt).
    month_year = (all_df["timestamp_month"] + all_df["timestamp_year"] * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    all_df["month_year_cnt"] = month_year.map(month_year_cnt_map)

    # Creating a column for week-year count
    week_year = (all_df["timestamp_weekofyear"] + all_df["timestamp_year"] * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    all_df["week_year_cnt"] = week_year.map(week_year_cnt_map)
    all_df.drop(['timestamp_weekofyear'], axis=1, inplace=True)

    return all_df
