#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 00:23:45 2024

@author: noahkhan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:01:51 2024

@author: noahkhan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 03:49:58 2024

@author: noahkhan
"""

import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import PlayerGameLog, playercareerstats, commonplayerinfo, leaguedashteamstats, scoreboardv2, boxscoretraditionalv2, commonteamroster,leaguedashplayerstats
from nba_api.stats.static import players
import plotly.express as px
import plotly.graph_objects as go
import datetime
from fuzzywuzzy import process
import math

def binomial_probability(k, n, p):
    return math.comb(n, k) * (p**k) * ((1-p)**(n-k))




def predict_player_stats_against_team(player_name, team_name, player_vs_team_df, last_10_games_df, team_stats_df):
    """
    A simple heuristic model:
    player_vs_team_df: DataFrame of player’s past games vs this team
    last_10_games_df: DataFrame of player’s last 10 games
    team_stats_df: DataFrame of league team stats (current season) to gauge team defense
    """

    if player_vs_team_df.empty or last_10_games_df.empty:
        # If we lack data, return None or some default
        return None

    # Compute player’s averages in last 10 games
    avg_pts_10 = last_10_games_df['Points'].mean()
    avg_reb_10 = last_10_games_df['Rebounds'].mean()
    avg_ast_10 = last_10_games_df['Assists'].mean()

    # Compute player’s historical vs team averages
    avg_pts_vs_team = player_vs_team_df['PTS'].mean()
    avg_reb_vs_team = player_vs_team_df['REB'].mean()
    avg_ast_vs_team = player_vs_team_df['AST'].mean()

    # Fetch the team's defensive strength
    # Let's say we use 'PTS' column from team_stats_df to see how many points they allow on average.
    # Actually, team_stats_df includes each team’s own stats, not exactly what they allow. 
    # For simplicity, assume a team with a lower 'PTS' rank is better defensively 
    # (In reality you'd use advanced metrics or allowed stats if available).
    
    # We can approximate defense by how low their PTS rank is:
    # If they rank highly in points (score a lot), that doesn't necessarily mean strong defense,
    # but let's say we do: A team that is low in PTS might be a slow-paced/defensive team.
    # This is a big assumption, but it's just a demo.

    team_row = team_stats_df[team_stats_df['TEAM_NAME'].str.lower() == team_name.lower()]
    if team_row.empty:
        # If no data for the team, no adjustment
        defense_adjustment_factor = 1.0
    else:
        # Assume AST_RANK or STL_RANK etc. can hint at defense. Let’s use 'PTS' rank (PT_RANK)
        pts_rank = team_row['PT_RANK'].values[0]  # rank in points
        # A lower rank number means a better offensive team, not necessarily defensive, but let's assume:
        # If the team is top 5 in PT_RANK (meaning they rank highly in points), 
        # we’ll say they might be weaker defensively (just as an example). 
        # Actually, let's invert logic: If PT_RANK is low (like 1 = best in PTS), 
        # maybe they're offensively strong, not necessarily defensive. 
        # Let's pick a different column to guess defense: 'TOV_RANK' or 'REB_RANK'.
        # Actually let's just do a neutral assumption:
        
        # If PT_RANK <= 5: strong team (we reduce player predicted stats by 5%)
        # If PT_RANK >= 25: weak team (increase by 5%)
        # Else no change.
        if pts_rank <= 5:
            defense_adjustment_factor = 0.95
        elif pts_rank >= 25:
            defense_adjustment_factor = 1.05
        else:
            defense_adjustment_factor = 1.0

    # Combine averages
    pred_points = ((avg_pts_10 + avg_pts_vs_team) / 2) * defense_adjustment_factor
    pred_rebounds = ((avg_reb_10 + avg_reb_vs_team) / 2) * defense_adjustment_factor
    pred_assists = ((avg_ast_10 + avg_ast_vs_team) / 2) * defense_adjustment_factor

    return {
        'Points': pred_points,
        'Rebounds': pred_rebounds,
        'Assists': pred_assists
    }




TEAM_ID_TO_NAME = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls",
    1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612754: "Indiana Pacers",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans",
    1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612764: "Washington Wizards",
}


TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

def real_time_game_data_tab():
    st.title("Real-Time NBA Game Data 🏀")

    # Introduction text
    st.markdown("""
    ### Features:
    - Select Yesterday's, Today's, or Tomorrow's games.
    - Live scoreboard with home and away teams and real-time scores.
    - Current game status, including quarter and time left.
    - Live player stats updates.
    """)

    # Add a radio button to select which day's games to view
    day_choice = st.radio(
        "Select which day's games to view:",
        ("Yesterday", "Today", "Tomorrow")
    )

    # Determine the selected date
    if day_choice == "Yesterday":
        selected_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    elif day_choice == "Tomorrow":
        selected_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # Today
        selected_date = datetime.datetime.now().strftime('%Y-%m-%d')

    try:
        # Fetch the scoreboard data for the selected date
        scoreboard = scoreboardv2.ScoreboardV2(game_date=selected_date)
        game_data = scoreboard.get_data_frames()

        if not game_data or game_data[0].empty:
            st.info("No NBA games scheduled for this date.")
            return

        # game_summary_df for basic game info
        game_summary_df = game_data[0]

        # line_score_df for detailed scoring info
        if len(game_data) > 1 and not game_data[1].empty:
            line_score_df = game_data[1]
        else:
            line_score_df = pd.DataFrame()

        # Loop through each game
        for _, game in game_summary_df.iterrows():
            game_id = game['GAME_ID']
            home_team_id = game['HOME_TEAM_ID']
            visitor_team_id = game['VISITOR_TEAM_ID']

            home_team = TEAM_ID_TO_NAME.get(home_team_id, "Unknown Team")
            visitor_team = TEAM_ID_TO_NAME.get(visitor_team_id, "Unknown Team")

            game_status = game['GAME_STATUS_TEXT']
            live_period = game.get('LIVE_PERIOD', 'N/A')
            live_pc_time = game.get('LIVE_PC_TIME', 'N/A')

            # Default score values
            pts_home = 'N/A'
            pts_visitor = 'N/A'

            # If we have line_score data, try to get final or current points from there
            if not line_score_df.empty:
                # Filter rows for this GAME_ID
                this_game_lines = line_score_df[line_score_df['GAME_ID'] == game_id]
                
                # Extract home/visitor scores if available
                home_line = this_game_lines[this_game_lines['TEAM_ID'] == home_team_id]
                visitor_line = this_game_lines[this_game_lines['TEAM_ID'] == visitor_team_id]
                
                if not home_line.empty:
                    pts_home = home_line['PTS'].values[0] if 'PTS' in home_line.columns else 'N/A'
                if not visitor_line.empty:
                    pts_visitor = visitor_line['PTS'].values[0] if 'PTS' in visitor_line.columns else 'N/A'

            # Display game details
            st.subheader(f"{home_team} vs {visitor_team}")
            st.write(f"**Game Status:** {game_status}")

            # Determine score line based on status
            if game_status.lower() == "in progress":
                score_line = f"{home_team} {pts_home} - {visitor_team} {pts_visitor}"
                st.write(f"**Current Quarter:** {live_period}")
                st.write(f"**Time Remaining:** {live_pc_time}")
            elif "final" in game_status.lower():
                # For final, we rely on line scores
                if pts_home == 'N/A' or pts_visitor == 'N/A':
                    # If still N/A, scoreboard may not have posted final points. 
                    # This can happen if the data isn't fully updated or no data returned.
                    score_line = f"Final Score: {home_team} N/A - {visitor_team} N/A"
                else:
                    score_line = f"Final Score: {home_team} {pts_home} - {visitor_team} {pts_visitor}"
            elif game_status.lower() == "scheduled":
                # If scheduled, no points yet. Assume 0-0 start.
                score_line = f"Scheduled: {home_team} 0 - 0 {visitor_team}"
                st.write(f"**Scheduled Start Time:** {game['GAME_TIME']}")
            else:
                # Other statuses (e.g., not started), default to 0-0
                if pts_home == 'N/A' and pts_visitor == 'N/A':
                    score_line = f"{home_team} 0 - 0 {visitor_team}"
                else:
                    score_line = f"{home_team} {pts_home} - {visitor_team} {pts_visitor}"

            st.write(f"**Score:** {score_line}")

            # Button for live stats
            if st.button(f"Live Updates for {home_team} vs {visitor_team}", key=f"update_{game_id}"):
                st.write("Fetching live stats...")
                # Fetch and display live player stats
                player_stats = fetch_live_player_stats(game_id)
                if not player_stats.empty:
                    st.write(f"### Live Stats for {home_team} vs {visitor_team}")
                    st.dataframe(player_stats)
                else:
                    st.write("No live stats available yet.")

            # Button for future win probability feature
            if st.button(f"Win Probability for {home_team} vs {visitor_team}", key=f"probability_{game_id}"):
                # Fetch team stats for current season
                team_stats_df = fetch_team_stats()
            
                # Call the predictive model
                est_a_score, est_b_score, wp_a = predict_future_game(home_team, visitor_team, team_stats_df)
            
                # Display the results
                st.write(f"**Predicted Score:** {home_team} {est_a_score:.1f} - {visitor_team} {est_b_score:.1f}")
                st.write(f"**Win Probability:** {home_team} {(wp_a*100):.1f}%, {visitor_team} {(100 - wp_a*100):.1f}%")


            # Separator between games
            st.write("---")

    except Exception as e:
        st.error(f"Error fetching live game data: {e}")

# Function to search for a player by name
@st.cache_data
def search_player(name):
    player = players.find_players_by_full_name(name)
    if not player:
        return "Player not found"
    return player[0]

@st.cache_data
def get_player_info(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        
        # Calculate "Years in League" based on career stats data
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        if not career_stats.empty:
            first_season = int(career_stats['SEASON_ID'].iloc[0][:4])  # Earliest season start year
            last_season = int(career_stats['SEASON_ID'].iloc[-1][:4])  # Latest season end year
            years_in_league = (last_season - first_season) + 1
        else:
            years_in_league = "N/A"

        overview = {
            "Full Name": player_info['DISPLAY_FIRST_LAST'].iloc[0],
            "Birthday": pd.Timestamp(player_info['BIRTHDATE'].iloc[0]).strftime("%B %d, %Y"),
            "Age": pd.Timestamp.now().year - pd.Timestamp(player_info['BIRTHDATE'].iloc[0]).year,
            "Years in League": years_in_league,  # Updated
            "Height": player_info['HEIGHT'].iloc[0],
            "Weight": player_info['WEIGHT'].iloc[0],
            "Position": player_info['POSITION'].iloc[0],
            "Draft Year": player_info['DRAFT_YEAR'].iloc[0],
            "Draft Round": player_info['DRAFT_ROUND'].iloc[0],
            "Draft Number": player_info['DRAFT_NUMBER'].iloc[0],
            "College": player_info['SCHOOL'].iloc[0] if player_info['SCHOOL'].iloc[0] else "N/A",
        }
        return overview
    except Exception as e:
        st.error(f"Error retrieving player info: {e}")
        return {}



# Function to get the career stats of a player
@st.cache_data
def get_career_stats(player_id):
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_df = career.get_data_frames()[0]

        # Ensure the SEASON column exists
        if 'SEASON' not in career_df.columns:
            career_df['SEASON'] = career_df['SEASON_ID'].apply(lambda x: f"{int(str(x)[:4])}-{int(str(x)[4:])}")

        # Remove unnecessary columns
        career_df = career_df.drop(columns=['PLAYER_ID', 'LEAGUE_ID', 'TEAM_ID'])
        
        # Calculate per-game stats
        career_df['MPG'] = (career_df['MIN'] / career_df['GP']).round(2)
        career_df['PPG'] = (career_df['PTS'] / career_df['GP']).round(2)
        career_df['RPG'] = (career_df['REB'] / career_df['GP']).round(2)
        career_df['APG'] = (career_df['AST'] / career_df['GP']).round(2)
        career_df['SPG'] = (career_df['STL'] / career_df['GP']).round(2)
        career_df['BPG'] = (career_df['BLK'] / career_df['GP']).round(2)
        career_df['FGM'] = (career_df['FGM'] / career_df['GP']).round(2)
        career_df['FGA'] = (career_df['FGA'] / career_df['GP']).round(2)
        career_df['FG%'] = career_df['FG_PCT'].round(2)
        career_df['3PM'] = (career_df['FG3M'] / career_df['GP']).round(2)
        career_df['3PA'] = (career_df['FG3A'] / career_df['GP']).round(2)
        career_df['3P%'] = career_df['FG3_PCT'].round(2)
        career_df['FTM'] = (career_df['FTM'] / career_df['GP']).round(2)
        career_df['FTA'] = (career_df['FTA'] / career_df['GP']).round(2)
        career_df['FT%'] = career_df['FT_PCT'].round(2)
        career_df['TS%'] = (career_df['PTS'] / (2 * (career_df['FGA'] + 0.44 * career_df['FTA']))).round(2)
        career_df['OREB'] = (career_df['OREB'] / career_df['GP']).round(2)
        career_df['DREB'] = (career_df['DREB'] / career_df['GP']).round(2)
        career_df['TOV'] = (career_df['TOV'] / career_df['GP']).round(2)

        # Include SEASON column and reorder columns
        column_order = [
            'SEASON', 'GP', 'GS', 'MPG', 'PPG', 'RPG', 'APG', 'SPG', 'BPG',
            'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'TS%',
            'OREB', 'DREB', 'TOV'
        ]
        career_df = career_df[column_order]

      # Calculate career averages
        career_avg = career_df.iloc[:, 1:].mean().round(2).to_frame().T
        career_avg.insert(0, 'SEASON', 'Career')

        # Calculate current season averages
        season_avg = career_df.iloc[[-1]].copy()
        season_avg['SEASON'] = 'This Season'

        return career_df, career_avg, season_avg

    except Exception as e:
        st.error(f"Error retrieving career stats: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()



# Function to get last N games trends
def get_last_10_games(player_id):
    try:
        gamelog = PlayerGameLog(player_id=player_id).get_data_frames()[0]
        gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])
        gamelog = gamelog.sort_values(by='GAME_DATE', ascending=False).head(10)

        # Ensure necessary columns exist
        if 'FG3M' in gamelog.columns and 'PTS' in gamelog.columns and 'REB' in gamelog.columns and 'AST' in gamelog.columns:
            gamelog['3 Points Made'] = gamelog['FG3M']
            gamelog['PRA'] = gamelog['PTS'] + gamelog['REB'] + gamelog['AST']
            gamelog['PA'] = gamelog['PTS'] + gamelog['AST']  # Points + Assists
            gamelog['PR'] = gamelog['PTS'] + gamelog['REB']  # Points + Rebounds
            gamelog['RA'] = gamelog['REB'] + gamelog['AST']  # Rebounds + Assists
           
        else:
            st.error("Missing necessary columns in game log data.")
            return pd.DataFrame()

        # Select and rename columns
        gamelog = gamelog[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', '3 Points Made', 'PRA', 'PA', 'RA', 'PR']]
        gamelog.columns = ['Date', 'Matchup', 'Points', 'Rebounds', 'Assists', '3 Points Made', 'PRA', 'PA', 'RA', 'PR']
        return gamelog

    except Exception as e:
        st.error(f"Error retrieving last 10 games: {e}")
        return pd.DataFrame()


def plot_bar_with_trends_and_table(gamelog, season_gamelog):
    st.write("### Trend Analysis with Hit Rates")

    # Select a category and trend lines
    category = st.selectbox("Select a Category:", ['Points', 'Rebounds', 'Assists', '3 Points Made', 'PRA', 'PA', 'RA', 'PR'])
    trend_values = st.text_input("Enter Trend Values (comma-separated):", value="15,20,25")
    trend_values = [float(val.strip()) for val in trend_values.split(",")]

    # Define hit rate threshold filter
    hit_rate_threshold = st.slider("Select Hit Rate Threshold (%)", min_value=0, max_value=100, value=50) / 100

    # Prepare bar graph
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gamelog['Date'],
        y=gamelog[category],
        name=category,
        marker_color='blue',
        hovertext=gamelog.apply(lambda row: f"Date: {row['Date']}<br>Opponent: {row['Matchup']}<br>{category}: {row[category]}", axis=1),
        hoverinfo="text"
    ))

    # Add trend lines
    for trend in trend_values:
        fig.add_trace(go.Scatter(
            x=gamelog['Date'],
            y=[trend] * len(gamelog),
            mode='lines',
            name=f"{category} ≥ {trend}",
            line=dict(dash='dash')
        ))

    # Show bar graph
    st.plotly_chart(fig)

    # Calculate hits for each trend
    results = []
    for trend in trend_values:
        hits = gamelog[category] >= trend
        season_hits = season_gamelog[category] >= trend
        hit_counts = [hits.iloc[:n].sum() / n if n <= len(hits) else 0 for n in [10, 5, 3]]
        results.append({
            "Trend": f"{category} ≥ {trend}",
            "Last 10": f"{hits.iloc[:10].sum()}/10 ({hit_counts[0]*100:.0f}%)",
            "Last 5": f"{hits.iloc[:5].sum()}/5 ({hit_counts[1]*100:.0f}%)",
            "Last 3": f"{hits.iloc[:3].sum()}/3 ({hit_counts[2]*100:.0f}%)",
            "Season": f"{season_hits.sum()}/{len(season_gamelog)} ({(season_hits.sum() / len(season_gamelog)) * 100:.0f}%)"
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Highlight results based on threshold
    def highlight_hit_rate(val):
        """Apply color formatting based on hit rate threshold."""
        if isinstance(val, str) and '/' in val:
            percentage = float(val.split('(')[-1].replace('%', '').replace(')', ''))
            color = 'green' if percentage >= hit_rate_threshold * 100 else 'red'
            return f'background-color: {color}; color: white;'
        return ''

    # Display table with formatting
    st.write("### Trend Analysis Table")
    styled_df = results_df.style.applymap(highlight_hit_rate, subset=['Last 10', 'Last 5', 'Last 3', 'Season'])
    st.dataframe(styled_df)



# Function to plot trends for career stats
def plot_category_trend(career_stats_df):
    st.write("### View Career Category Trends")
    
    # Determine valid default values based on available columns
    default_categories = ['PPG', 'RPG', 'APG']  # Possible default values
    available_defaults = [cat for cat in default_categories if cat in career_stats_df.columns[3:]]

    # Dropdown for selecting categories
    categories = st.multiselect(
        "Select up to 3 Categories:",
        options=career_stats_df.columns[2:],  # Exclude non-numeric columns
        default=available_defaults[:2]  # Use available defaults
    )

    if categories:
        fig = px.line(
            career_stats_df.melt(id_vars=['SEASON'], value_vars=categories, var_name='Category', value_name='Value'),
            x='SEASON',
            y='Value',
            color='Category',
            title=f"Trends Over Career for Selected Categories",
            labels={"SEASON": "Season", "Value": "Stat Value"},
            markers=True
        )
        st.plotly_chart(fig)

        # Allow download of graph data as CSV
        csv_data = career_stats_df[['SEASON'] + categories].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Selected Categories Data as CSV",
            data=csv_data,
            file_name=f"selected_categories_trend_data.csv",
            mime='text/csv'
        )

    # Button for showing all categories
    if st.button("Show All Categories"):
        fig_all = px.line(
            career_stats_df.melt(id_vars=['SEASON'], value_vars=career_stats_df.columns[3:], var_name='Category', value_name='Value'),
            x='SEASON',
            y='Value',
            color='Category',
            title="Trends Over Career for All Categories",
            labels={"SEASON": "Season", "Value": "Stat Value"},
            markers=True
        )
        st.plotly_chart(fig_all)


def compare_players(player1_name, player2_name):
    try:
        # Search for players
        player1 = search_player(player1_name)
        player2 = search_player(player2_name)

        if isinstance(player1, str) or isinstance(player2, str):
            st.error("One or both players not found.")
            return pd.DataFrame(), pd.DataFrame(), "Error: Player not found."

        # Fetch career and season stats for both players
        _, player1_career_avg, player1_season_avg = get_career_stats(player1['id'])
        _, player2_career_avg, player2_season_avg = get_career_stats(player2['id'])

        if player1_career_avg.empty or player2_career_avg.empty:
            st.error("No career stats available for one or both players.")
            return pd.DataFrame(), pd.DataFrame(), "Error: No stats available."

        # Prepare data for tables (transpose)
        season_df = pd.DataFrame({
            "Stat": player1_season_avg.columns[1:],
            player1_name: player1_season_avg.iloc[0, 1:].values,
            player2_name: player2_season_avg.iloc[0, 1:].values,
        }).set_index("Stat").T  # Transpose to make players rows

        career_df = pd.DataFrame({
            "Stat": player1_career_avg.columns[1:],
            player1_name: player1_career_avg.iloc[0, 1:].values,
            player2_name: player2_career_avg.iloc[0, 1:].values,
        }).set_index("Stat").T  # Transpose to make players rows

        return season_df, career_df, None

    except Exception as e:
        st.error(f"Error comparing players: {e}")
        return pd.DataFrame(), pd.DataFrame(), f"Error: {e}"



# Function to calculate Games Back (GB)
def calculate_games_back(team_stats_df):
    """Calculate Games Back (GB) for each team in the conference."""
    team_stats_df['GB'] = 0.0
    for conference in team_stats_df['CONFERENCE'].unique():
        conf_teams = team_stats_df[team_stats_df['CONFERENCE'] == conference]
        max_wins = conf_teams['W'].max()
        min_losses = conf_teams['L'].min()
        team_stats_df.loc[team_stats_df['CONFERENCE'] == conference, 'GB'] = (
            ((max_wins - team_stats_df['W']) + (team_stats_df['L'] - min_losses)) / 2
        )
    return team_stats_df


# Function to generate formatted record strings
def format_record(wins, losses):
    return f"{wins}-{losses}"


@st.cache_data
def fetch_team_stats(season='2024-25'):
    # Initialize to an empty DataFrame so it's defined even if we hit an exception.
    team_stats_df = pd.DataFrame()
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        team_stats_df = team_stats.get_data_frames()[0]

        # Map teams to conferences (as before)
        eastern_teams = [
            "Boston Celtics", "Milwaukee Bucks", "Philadelphia 76ers", "Brooklyn Nets",
            "New York Knicks", "Cleveland Cavaliers", "Atlanta Hawks", "Miami Heat",
            "Chicago Bulls", "Indiana Pacers", "Orlando Magic", "Toronto Raptors",
            "Charlotte Hornets", "Washington Wizards", "Detroit Pistons"
        ]

        team_stats_df['CONFERENCE'] = team_stats_df['TEAM_NAME'].apply(
            lambda x: "East" if x in eastern_teams else "West"
        )

        # Calculate GB and ranks as before
        team_stats_df['GB'] = team_stats_df.groupby('CONFERENCE')['W'].transform(
            lambda x: (max(x) - x) / 2
        )

        ranking_fields = {
            "FG_PCT": "FG_PCT_RANK",
            "FG3_PCT": "G3_PCT_RANK",
            "PTS": "PT_RANK",
            "REB": "REB_RANK",
            "AST": "AST_RANK",
            "TOV": "TOV_RANK",
            "STL": "STL_RANK",
            "BLK": "BLK_RANK"
        }

        for stat_field, rank_field in ranking_fields.items():
            if stat_field in team_stats_df.columns:
                team_stats_df[rank_field] = team_stats_df[stat_field].rank(ascending=False, method='min')

        columns_order = [
            'TEAM_NAME', 'CONFERENCE', 'GP', 'W', 'L', 'GB', 'W_PCT',
            'FG_PCT', 'FG_PCT_RANK', 'FG3_PCT', 'G3_PCT_RANK',
            'PTS', 'PT_RANK', 'REB', 'REB_RANK',
            'AST', 'AST_RANK', 'TOV', 'TOV_RANK',
            'STL', 'STL_RANK', 'BLK', 'BLK_RANK'
        ]

        team_stats_df = team_stats_df[columns_order]

        return team_stats_df

    except Exception as e:
        st.error(f"Error fetching team stats for {season}: {e}")
        # Return an empty DataFrame to avoid referencing undefined variables
        return pd.DataFrame()

    
@st.cache_data
def fetch_live_player_stats(game_id):
    try:
        # Fetch the box score data
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        raw_player_stats = boxscore.get_data_frames()[0]

        # Define required columns
        required_columns = ["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
        
        # Check for available columns in the data
        available_columns = [col for col in required_columns if col in raw_player_stats.columns]
        
        if not available_columns:
            st.error("No relevant stats found in the live player stats response.")
            return pd.DataFrame()
        
        # Filter for available columns only
        player_stats = raw_player_stats[available_columns]

        # Rename columns for readability
        rename_mapping = {
            "PLAYER_NAME": "Player",
            "TEAM_ABBREVIATION": "Team",
            "MIN": "Minutes",
            "PTS": "Points",
            "REB": "Rebounds",
            "AST": "Assists",
            "STL": "Steals",
            "BLK": "Blocks",
            "TOV": "Turnovers"
        }
        player_stats.rename(columns={col: rename_mapping[col] for col in available_columns if col in rename_mapping}, inplace=True)

        # Fill missing stats with 0
        player_stats.fillna(0, inplace=True)

        return player_stats

    except Exception as e:
        st.error(f"Error fetching live player stats: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_team_roster(team_name, season='2024-25'):
    try:
        # Get team ID based on name
        team_id = None
        for key, value in TEAM_ID_TO_NAME.items():
            if value.lower() == team_name.lower():
                team_id = key
                break

        if not team_id:
            st.error("Team not found. Please check the name and try again.")
            return pd.DataFrame()

        # Fetch roster data for the given season
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        roster_df = roster.get_data_frames()[0]

        # Fetch player stats for the given season
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]

        # Merge roster and player stats
        roster_stats = pd.merge(
            roster_df,
            player_stats,
            how="left",
            left_on="PLAYER_ID",
            right_on="PLAYER_ID"
        )

        # Add calculated per-game averages
        # Guard against division by zero in case GP is zero
        roster_stats["PPG"] = (roster_stats["PTS"] / roster_stats["GP"]).round(2).fillna(0)
        roster_stats["APG"] = (roster_stats["AST"] / roster_stats["GP"]).round(2).fillna(0)
        roster_stats["RPG"] = (roster_stats["REB"] / roster_stats["GP"]).round(2).fillna(0)

        # Filter columns for the table
        table_columns = [
            "PLAYER",
            "POSITION",
            "HEIGHT",
            "PPG",
            "APG",
            "RPG"
        ]
        roster_stats = roster_stats[table_columns]

        # Rename columns
        roster_stats.rename(columns={
            "PLAYER": "Player",
            "POSITION": "Position",
            "HEIGHT": "Height",
            "PPG": "Points Per Game",
            "APG": "Assists Per Game",
            "RPG": "Rebounds Per Game"
        }, inplace=True)

        # Sort by PPG to approximate starting 5
        roster_stats.sort_values(by="Points Per Game", ascending=False, inplace=True)

        # Add column for starter (top 5 scorers)
        roster_stats.reset_index(drop=True, inplace=True)
        roster_stats["Starter"] = ["Yes" if i < 5 else "No" for i in range(len(roster_stats))]

        return roster_stats

    except Exception as e:
        st.error(f"Error fetching team roster for season {season}: {e}")
        return pd.DataFrame()


def fetch_team_historical_data(team1_name, team2_name):
    # Simulated function to fetch historical head-to-head data
    # Replace with actual logic to fetch from NBA API or data source
    data = {
        "Date": ["2024-11-10", "2024-01-15"],
        "Team 1": [team1_name, team1_name],
        "Team 2": [team2_name, team2_name],
        "Team 1 Points": [102, 115],
        "Team 2 Points": [98, 120],
    }
    return pd.DataFrame(data)

@st.cache_data
def fetch_player_vs_team_data(player_id, team_name):
    try:
        team_name = team_name.title()
        team_abbr = TEAM_NAME_TO_ABBR.get(team_name)
        
        if not team_abbr:
            st.error(f"Team '{team_name}' not found in abbreviation mapping. Please update TEAM_NAME_TO_ABBR.")
            return pd.DataFrame()

        gamelog = PlayerGameLog(player_id=player_id).get_data_frames()[0]

        if "MATCHUP" not in gamelog.columns:
            st.error("MATCHUP column not found in game log data.")
            return pd.DataFrame()

        filtered_logs = gamelog[gamelog['MATCHUP'].str.contains(team_abbr, na=False)]

        if filtered_logs.empty:
            st.info(f"No games found for player vs. {team_name}.")
            return pd.DataFrame()

        filtered_logs["Home/Away"] = filtered_logs["MATCHUP"].apply(
            lambda x: "Home" if "vs." in x else "Away"
        )

        # Remove unwanted columns if they exist
        columns_to_remove = ["SEASON_ID", "PLAYER_ID", "GAME_ID", "VIDEO_AVAILABLE"]
        for col in columns_to_remove:
            if col in filtered_logs.columns:
                filtered_logs.drop(columns=[col], inplace=True)

        return filtered_logs

    except Exception as e:
        st.error(f"Error fetching player vs. team data: {e}")
        return pd.DataFrame()


def get_season_gamelog(player_id):
    try:
        gamelog = PlayerGameLog(player_id=player_id).get_data_frames()[0]
        gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])
        gamelog = gamelog.sort_values(by='GAME_DATE', ascending=False)

        # Ensure necessary columns exist
        if 'FG3M' in gamelog.columns and 'PTS' in gamelog.columns and 'REB' in gamelog.columns and 'AST' in gamelog.columns:
            gamelog['3 Points Made'] = gamelog['FG3M']
            gamelog['PRA'] = gamelog['PTS'] + gamelog['REB'] + gamelog['AST']
            gamelog['PA'] = gamelog['PTS'] + gamelog['AST']
            gamelog['PR'] = gamelog['PTS'] + gamelog['REB']
            gamelog['RA'] = gamelog['REB'] + gamelog['AST']
        else:
            st.error("Missing necessary columns in season game log data.")
            return pd.DataFrame()

        # Select relevant columns
        gamelog = gamelog[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', '3 Points Made', 'PRA', 'PA', 'RA', 'PR']]
        gamelog.columns = ['Date', 'Matchup', 'Points', 'Rebounds', 'Assists', '3 Points Made', 'PRA', 'PA', 'RA', 'PR']
        return gamelog

    except Exception as e:
        st.error(f"Error retrieving season game log: {e}")
        return pd.DataFrame()

# Function to handle team name matching
def match_team_name(input_name, team_dict):
    """
    Matches an input team name to the closest name in the team dictionary.
    Returns the matched team name and its abbreviation.
    """
    team_names = list(team_dict.keys())
    closest_match, confidence = process.extractOne(input_name, team_names)
    if confidence >= 80:  # Confidence threshold
        return closest_match, team_dict[closest_match]
    else:
        return None, None

def predict_future_game(teamA_name, teamB_name, team_stats_df, league_avg_ppg=112):
    # Extract season stats for teams (PPG)
    teamA_data = team_stats_df[team_stats_df['TEAM_NAME'].str.lower() == teamA_name.lower()].iloc[0]
    teamB_data = team_stats_df[team_stats_df['TEAM_NAME'].str.lower() == teamB_name.lower()].iloc[0]

    TeamA_PPG = teamA_data['PTS'] / teamA_data['GP']  # Points per game
    # Simple approximation for PAPG just using current season W_PCT
    TeamA_PAPG = TeamA_PPG + (10 * (1 - teamA_data['W_PCT']))

    TeamB_PPG = teamB_data['PTS'] / teamB_data['GP']
    TeamB_PAPG = TeamB_PPG + (10 * (1 - teamB_data['W_PCT']))

    # Since we're only using current season data, no historical adjustments:
    HeadToHead_Adjustment_A = 0
    HeadToHead_Adjustment_B = 0

    # Simple model using only current season data
    Off_Str_A = TeamA_PPG - league_avg_ppg
    Def_Str_B = league_avg_ppg - TeamB_PPG
    Est_A_Score = TeamA_PPG + (Off_Str_A * 0.5) - (Def_Str_B * 0.5) + HeadToHead_Adjustment_A

    Off_Str_B = TeamB_PPG - league_avg_ppg
    Def_Str_A = league_avg_ppg - TeamA_PPG
    Est_B_Score = TeamB_PPG + (Off_Str_B * 0.5) - (Def_Str_A * 0.5) + HeadToHead_Adjustment_B

    Point_Diff = Est_A_Score - Est_B_Score

    import math
    C = 0.1
    WP_A = 1 / (1 + math.exp(-C * Point_Diff))

    return Est_A_Score, Est_B_Score, WP_A



# Main Streamlit app
def main():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Player Profile", "Compare", "Real-Time Game Data", "Team Stats"])
    
    with tab1:
        st.title("🏀 Welcome to the NBA Player Analytics App!")
        st.markdown("""
        <style>
            .highlight {
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;
                color: #003366;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        ## 📝 About This App
        This application provides powerful analytics for NBA players, catering to basketball enthusiasts, fantasy sports players, and data analysts alike.
        
        ---   
        ## Key Features
        
        ### **Player Profile Analysis**  
        - Get **detailed career stats**, season averages, and performance trends for any NBA player.  
        - Visualize a player's recent performance over the **last 10 games**.  
        - **Download player stats** for further analysis.  
    
        ### **Player Comparison**  
        - Compare two players side-by-side to evaluate their **strengths and weaknesses**.  
        - Visualize **key stats** to identify standout performers.  
    
        ### **Real-Time Game Data**  
        - View **live NBA game scores** and key metadata **updated in real-time**.  
        - Explore today’s games, including **scores**, game status, quarter-by-quarter updates, and time remaining.  
        - Track **live point updates** and **win probabilities** for ongoing games.  
        ---
        ## How to Use  
        - Navigate through the app using the **tabs**.  
        - Search for players by name with the **auto-complete feature**.  
        - Compare players or explore detailed analytics for individual players.  
        - Check **live game data** to stay updated on ongoing NBA games, including scoring and real-time updates.  
    
        ---
        ## Data Sources  
        - Player data is sourced from the **[NBA API](https://www.nba.com/stats/)**.  
        - Historical data scraped from **[Basketball Reference](https://www.basketball-reference.com/)**.  
        - Real-time game data retrieved from the **NBA API's Scoreboard V2 Endpoint**.  
    
        ---
        **Feel free to reach out with feedback or suggestions to make this app even better. Enjoy exploring the world of NBA analytics!**  
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/NBA_logo.svg/1920px-NBA_logo.svg.png", 
                 width=300)

    # Tab 2: Player Profile
    with tab2:
        st.title("Player Profile")
        name = st.text_input("Enter the player's name (e.g., LeBron James):")
        if name:
            player = search_player(name)
            if isinstance(player, str):
                st.error(player)
                return

            try:
                player_info = get_player_info(player['id'])
                if player_info:
                    st.write("### Player Overview")
                    overview_df = pd.DataFrame(player_info, index=[0]).T.reset_index()
                    overview_df.columns = ['Attribute', 'Value']
                    st.table(overview_df)

                career_stats_df, career_avg, season_avg = get_career_stats(player['id'])
                if not career_stats_df.empty:
                    st.write(f"### Career Stats for {player['full_name']} (Per Game):")
                    st.write(career_stats_df)
                    st.write(f"### Career Averages for {player['full_name']}")
                    st.table(career_avg)
                    st.write(f"### Season Averages for {player['full_name']}")
                    st.table(season_avg)

                    csv_data = career_stats_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Career Stats as CSV",
                        data=csv_data,
                        file_name=f"{player['full_name'].replace(' ', '_')}_career_stats.csv",
                        mime='text/csv'
                    )
                    plot_category_trend(career_stats_df)

                # Inside the Player Profile Tab
                last_10_games = get_last_10_games(player['id'])
                season_gamelog = get_season_gamelog(player['id'])  # Fetch full season data
                
                if not last_10_games.empty and not season_gamelog.empty:
                    st.write("### Last 10 Games Stats")
                    st.write(last_10_games)
                    plot_bar_with_trends_and_table(last_10_games, season_gamelog)

            except Exception as e:
                st.error(f"Error retrieving data: {e}")

   # Tab 3: Compare Players, Teams, or Players vs. Teams
    with tab3:
        st.title("Compare 🏀")
        
        # Dropdown to select comparison type
        compare_type = st.selectbox(
            "Choose comparison type:",
            ["Player vs. Player", "Team vs. Team", "Player vs. Team"]
        )
        
        # Player vs. Player Comparison
        if compare_type == "Player vs. Player":
            st.subheader("Player vs. Player Comparison")
            player1_name = st.text_input("Enter the first player's name:")
            player2_name = st.text_input("Enter the second player's name:")
        
            if player1_name and player2_name:
                player1 = search_player(player1_name)
                player2 = search_player(player2_name)
        
                if isinstance(player1, str) or isinstance(player2, str):
                    st.error("One or both players not found.")
                else:
                    # Compare season and career stats as before
                    season_comparison, career_comparison, error = compare_players(player1_name, player2_name)
                    if error:
                        st.error(error)
                    else:
                        # Display season stats comparison
                        st.write(f"### Season Stats Comparison: {player1_name} vs {player2_name}")
                        st.dataframe(season_comparison)
        
                        # Display career stats comparison
                        st.write(f"### Career Stats Comparison: {player1_name} vs {player2_name}")
                        st.dataframe(career_comparison)
        
                        # --- NEW SECTION STARTS HERE ---
                        # 1. Get each player's current team info
                        try:
                            player1_info = commonplayerinfo.CommonPlayerInfo(player_id=player1['id']).get_data_frames()[0]
                            player2_info = commonplayerinfo.CommonPlayerInfo(player_id=player2['id']).get_data_frames()[0]
        
                            # Extract current team ID
                            player1_team_id = player1_info['TEAM_ID'].iloc[0]
                            player2_team_id = player2_info['TEAM_ID'].iloc[0]
        
                            player1_team_name = TEAM_ID_TO_NAME.get(player1_team_id, None)
                            player2_team_name = TEAM_ID_TO_NAME.get(player2_team_id, None)
        
                            if player1_team_name and player2_team_name:
                                # 2. Fetch player's game logs vs. the other player's current team
                                player1_vs_team = fetch_player_vs_team_data(player_id=player1['id'], team_name=player2_team_name)
                                player2_vs_team = fetch_player_vs_team_data(player_id=player2['id'], team_name=player1_team_name)
        
                                # 3. Display the tables
                                if not player1_vs_team.empty:
                                    st.write(f"### {player1_name} vs. {player2_team_name}")
                                    st.dataframe(player1_vs_team)
                                else:
                                    st.write(f"No data available for {player1_name} vs. {player2_team_name}")
        
                                if not player2_vs_team.empty:
                                    st.write(f"### {player2_name} vs. {player1_team_name}")
                                    st.dataframe(player2_vs_team)
                                else:
                                    st.write(f"No data available for {player2_name} vs. {player1_team_name}")
                            else:
                                st.write("Could not find one or both players' current team information.")
                        except Exception as e:
                            st.error(f"Error retrieving player team info or player vs team data: {e}")
                        # --- NEW SECTION ENDS HERE ---

        
        # Team vs. Team Comparison
        if compare_type == "Team vs. Team":
            st.subheader("Team vs. Team Comparison")
            team1_name = st.text_input("Enter the first team's name (e.g., 'Chicago Bulls'):")
            team1_year = st.number_input("Enter the year of the first team (e.g., 1996):", min_value=1947, max_value=2030, value=2025)
            team2_name = st.text_input("Enter the second team's name (e.g., 'Golden State Warriors'):")
            team2_year = st.number_input("Enter the year of the second team (e.g., 2016):", min_value=1947, max_value=2030, value=2025)

            if team1_name and team1_year and team2_name and team2_year:
                def year_to_season_str(year):
                    start = year - 1
                    end = str(year)[-2:]
                    return f"{start}-{end}"

                team1_season_str = year_to_season_str(team1_year)
                team2_season_str = year_to_season_str(team2_year)

                team1_name_matched, team1_abbr = match_team_name(team1_name, TEAM_NAME_TO_ABBR)
                team2_name_matched, team2_abbr = match_team_name(team2_name, TEAM_NAME_TO_ABBR)

                if not team1_name_matched or not team2_name_matched:
                    st.error("One or both team names could not be matched. Please check your input.")
                else:
                    st.write(f"Matched Teams: {team1_name} → {team1_name_matched}, {team2_name} → {team2_name_matched}")

                    # Fetch team stats for each chosen season
                    team1_stats_df = fetch_team_stats(season=team1_season_str)
                    team2_stats_df = fetch_team_stats(season=team2_season_str)

                    if team1_stats_df.empty:
                        st.error(f"No team stats found for the {team1_season_str} season for {team1_name_matched}.")
                    elif team2_stats_df.empty:
                        st.error(f"No team stats found for the {team2_season_str} season for {team2_name_matched}.")
                    else:
                        team1_stats = team1_stats_df[team1_stats_df['TEAM_NAME'].str.lower() == team1_name_matched.lower()]
                        team2_stats = team2_stats_df[team2_stats_df['TEAM_NAME'].str.lower() == team2_name_matched.lower()]

                        if team1_stats.empty:
                            st.error(f"{team1_name_matched} not found in the {team1_season_str} season data.")
                        elif team2_stats.empty:
                            st.error(f"{team2_name_matched} not found in the {team2_season_str} season data.")
                        else:
                            # Display team stats comparison
                            st.write(f"### {team1_season_str} {team1_name_matched} vs {team2_season_str} {team2_name_matched}")
                            comparison_data = pd.concat([team1_stats, team2_stats])
                            st.dataframe(comparison_data)

                            # Fetch and display rosters
                            roster_team1 = fetch_team_roster(team1_name_matched, season=team1_season_str)
                            roster_team2 = fetch_team_roster(team2_name_matched, season=team2_season_str)

                            if not roster_team1.empty:
                                st.write(f"### {team1_season_str} {team1_name_matched} Roster")
                                st.dataframe(roster_team1)
                            else:
                                st.info(f"No roster data available for {team1_season_str} {team1_name_matched}.")

                            if not roster_team2.empty:
                                st.write(f"### {team2_season_str} {team2_name_matched} Roster")
                                st.dataframe(roster_team2)
                            else:
                                st.info(f"No roster data available for {team2_season_str} {team2_name_matched}.")

                            # --- Insert prediction code here ---
                            # Use the predict_future_game function to get a single-game probability.
                            # We'll assume the same probability applies to all 7 games in the series.
                            # First, we need both teams' stats from their respective DataFrames:
                            # We already have team1_stats_df and team2_stats_df from above.

                            # Extract current season team_stats_df for each team. 
                            # Actually, we need to combine these teams into one df. Or we can just call fetch_team_stats
                            # again for each season and pick the team rows.

                            # For simplicity, we can just re-fetch team_stats for each season and combine:
                            # But we already have the needed data in team1_stats_df and team2_stats_df.
                            # We'll combine them into one DataFrame:
                            combined_team_stats = pd.concat([team1_stats, team2_stats])

                            # We'll just call predict_future_game with the matched team names and combined_team_stats.
                            est_a_score, est_b_score, wp_a = predict_future_game(team1_name_matched, team2_name_matched, combined_team_stats)

                            # Now simulate best-of-7:
                            # Probability Team A (team1) wins at least 4 out of 7:
                            p = wp_a
                            p_teamA_wins_series = 0
                            for k in range(4, 8):
                                p_teamA_wins_series += binomial_probability(k, 7, p)

                            st.write("### Best-of-7 Series Prediction")
                            st.write(f"Single game predicted score: {team1_name_matched} {est_a_score:.1f} - {team2_name_matched} {est_b_score:.1f}")
                            st.write(f"Single game win probability for {team1_name_matched}: {p*100:.1f}%")

                            if p_teamA_wins_series > 0.5:
                                st.write(f"**Predicted to win the 7-game series:** {team1_name_matched} (with {(p_teamA_wins_series*100):.1f}% probability)")
                            else:
                                st.write(f"**Predicted to win the 7-game series:** {team2_name_matched} (with {((1-p_teamA_wins_series)*100):.1f}% probability)")
        

        
        
        
        # Player vs. Team Comparison
        if compare_type == "Player vs. Team":
            st.subheader("Player vs. Team Comparison")
            player_name = st.text_input("Enter the player's name:")
            team_name = st.text_input("Enter the team's name (e.g., 'Boston Celtics'):")

            if player_name and team_name:
                player = search_player(player_name)
                if isinstance(player, str):
                    st.error(player)
                else:
                    matched_team_name, team_abbr = match_team_name(team_name, TEAM_NAME_TO_ABBR)
                    if not matched_team_name:
                        st.error("Could not match the team name. Please try a different or more complete name.")
                    else:
                        # Use the matched team name instead of the raw input
                        player_vs_team_df = fetch_player_vs_team_data(player['id'], matched_team_name)
                        if player_vs_team_df.empty:
                            st.info(f"No data available for {player_name} vs. {matched_team_name}.")
                        else:
                            st.write(f"### Performance of {player_name} Against {matched_team_name}")
                            st.table(player_vs_team_df)

                            # Fetch last 10 games data
                            last_10_games = get_last_10_games(player['id'])
                            if last_10_games.empty:
                                st.info("No last 10 games data available.")
                            else:
                                st.write("### Player's Last 10 Games")
                                st.table(last_10_games)

                                # Fetch team stats for the current season
                                team_stats_df = fetch_team_stats()

                                # Now call the prediction function
                                predicted_stats = predict_player_stats_against_team(player_name, matched_team_name, player_vs_team_df, last_10_games, team_stats_df)

                                if predicted_stats:
                                    st.write("### Predicted Stats for Next Game vs. This Team")
                                    st.write(f"Points: {predicted_stats['Points']:.1f}")
                                    st.write(f"Rebounds: {predicted_stats['Rebounds']:.1f}")
                                    st.write(f"Assists: {predicted_stats['Assists']:.1f}")
                                else:
                                    st.info("Not enough data to predict player stats.")

  

 # Tab 4: Real-Time Game Data
    with tab4:
        real_time_game_data_tab()
    
        try:
            # Fetch today's date
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Fetch live game data
            scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
            game_data = scoreboard.get_data_frames()
    
            if game_data and not game_data[0].empty:
                game_summary_df = game_data[0]  # Extract game summary
    
                for _, game in game_summary_df.iterrows():
                    game_id = game.get('GAME_ID')
                    home_team_id = game.get('HOME_TEAM_ID')
                    visitor_team_id = game.get('VISITOR_TEAM_ID')
                    home_team = TEAM_ID_TO_NAME.get(home_team_id, "N/A")
                    visitor_team = TEAM_ID_TO_NAME.get(visitor_team_id, "N/A")
                    pts_home = game.get('PTS_HOME', 'N/A')
                    pts_visitor = game.get('PTS_VISITOR', 'N/A')
                    game_status = game.get('GAME_STATUS_TEXT', 'N/A')
    
                    # Display game details
                    st.subheader(f"{home_team} vs {visitor_team}")
                    st.write(f"**Status:** {game_status}")
                    if game_status.lower() == "in progress":
                        st.write(f"**Score:** {home_team} {pts_home} - {visitor_team} {pts_visitor}")
    
                    # Button for live stats
                    if st.button(f"Live Updates for {home_team} vs {visitor_team}", key=f"update_{game_id}"):
                        st.write("Fetching live stats...")
                        # Fetch and display live player stats
                        player_stats = fetch_live_player_stats(game_id)
                        if not player_stats.empty:
                            st.write(f"### Live Stats for {home_team} vs {visitor_team}")
                            st.dataframe(player_stats)
                        else:
                            st.write("No live stats available yet.")
    
                   # Button for win probability
                    if st.button(f"Win Probability for {home_team} vs {visitor_team}", key=f"probability_{game_id}"):
                        # Remove the original 'not yet implemented' line and add model code instead
                        
                        # Fetch team stats for current season
                        team_stats_df = fetch_team_stats()  
                        
                        # Call the predictive model
                        est_a_score, est_b_score, wp_a = predict_future_game(home_team, visitor_team, team_stats_df)
                        
                        # Display the results
                        st.write(f"**Predicted Score:** {home_team} {est_a_score:.1f} - {visitor_team} {est_b_score:.1f}")
                        st.write(f"**Win Probability:** {home_team} {(wp_a*100):.1f}%, {visitor_team} {(100 - wp_a*100):.1f}%")

    
                    st.write("---")  # Separator between games
    
            else:
                st.info("No NBA games scheduled for today.")
    
        except Exception as e:
            st.error(f"Error fetching live game data: {e}")

       # Tab 5: Team Stats
    with tab5:
        st.title("NBA Team Stats 🏀")
        
        # Team stats ranking table
        team_stats_df = fetch_team_stats()
        if not team_stats_df.empty:
            conference = st.selectbox("Select Conference:", ["All", "East", "West"])
            if conference != "All":
                filtered_stats = team_stats_df[team_stats_df['CONFERENCE'] == conference]
            else:
                filtered_stats = team_stats_df
    
            st.write(f"### {conference} Conference Team Rankings" if conference != "All" else "### All Teams Rankings")
            st.dataframe(filtered_stats)
    
            # Team roster section
            team_name = st.text_input("Enter a Team Name to View Their Roster (e.g., 'Boston Celtics'):")
            if team_name:
                roster_stats = fetch_team_roster(team_name)
                if not roster_stats.empty:
                    st.write(f"### {team_name} Roster (2024-25 Season)")
                    st.dataframe(roster_stats)
    
                    # Allow downloading roster as CSV
                    csv_data = roster_stats.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download {team_name} Roster as CSV",
                        data=csv_data,
                        file_name=f"{team_name.replace(' ', '_')}_roster.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No team stats available at the moment.")

if __name__ == "__main__":
    main()
