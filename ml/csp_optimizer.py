"""
Constraint Satisfaction Problem (CSP) Optimizer for FPL Squad Selection

This module implements a CSP solver using Greedy Search
to select an optimal 15-player squad that maximizes predicted points while satisfying:
- 15 players total
- £100m budget constraint
- Position constraints (2 GK, 5 DEF, 5 MID, 3 FWD)
- Max 3 players per club
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from dataclasses import dataclass

@dataclass
class SquadConstraints:
    total_players: int = 15
    budget: float = 100.0
    positions: Dict[str, int] = None
    max_per_club: int = 3
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

@dataclass
class SquadSolution:
    player_ids: List[int]
    total_cost: float
    total_points: float
    position_distribution: Dict[str, int]
    club_distribution: Dict[str, int]
    is_valid: bool
    violations: List[str]

class CSPOptimizer:
    def __init__(self, constraints: Optional[SquadConstraints] = None):
        self.constraints = constraints or SquadConstraints()
    
    def validate_solution(self, player_ids: List[int], df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if a squad meets all rules (Budget, Team limits, Positions).
        """
        violations = []
        squad_df = df[df['id'].isin(player_ids)]
        
        # 1. Player Count
        if len(player_ids) != self.constraints.total_players:
            violations.append(f"Player count {len(player_ids)} != {self.constraints.total_players}")
        
        # 2. Budget
        total_cost = squad_df['cost'].sum()
        if total_cost > self.constraints.budget:
            violations.append(f"Budget exceeded: {total_cost:.1f} > {self.constraints.budget}")
            
        # 3. Positions
        pos_counts = squad_df['position'].value_counts()
        for pos, limit in self.constraints.positions.items():
            if pos_counts.get(pos, 0) != limit:
                violations.append(f"Wrong count for {pos}: {pos_counts.get(pos, 0)} != {limit}")
                
        # 4. Club Limits
        club_counts = squad_df['team'].value_counts()
        if (club_counts > self.constraints.max_per_club).any():
            violations.append("Too many players from one club")
            
        return len(violations) == 0, violations
    
    def optimize_squad(self, df: pd.DataFrame, predicted_points: Dict[int, float]) -> SquadSolution:
        """
        Greedy optimization to find best squad.
        """
        # Prepare Data
        df_opt = df.copy()
        
        # Map predictions to dataframe
        df_opt['pred_points'] = df_opt['id'].map(predicted_points).fillna(0)
        
        # Heuristic: Points per Million (Value)
        # We add a small epsilon to cost to avoid division by zero
        df_opt['value_ratio'] = df_opt['pred_points'] / df_opt['cost'].replace(0, 0.1)
        
        # Sort by value (best deals first)
        df_opt = df_opt.sort_values('value_ratio', ascending=False)
        
        selected_ids = []
        current_cost = 0.0
        current_pos = {k: 0 for k in self.constraints.positions}
        current_clubs = {}
        
        # --- PHASE 1: GREEDY SELECTION (High Value Players) ---
        for _, player in df_opt.iterrows():
            if len(selected_ids) == self.constraints.total_players: break
            
            pid = player['id']
            pos = player['position']
            team = player['team']
            cost = player['cost']
            
            # Check constraints
            if current_cost + cost > self.constraints.budget: continue
            if current_pos.get(pos, 0) >= self.constraints.positions.get(pos, 0): continue
            if current_clubs.get(team, 0) >= self.constraints.max_per_club: continue
            
            # Select Player
            selected_ids.append(pid)
            current_cost += cost
            current_pos[pos] = current_pos.get(pos, 0) + 1
            current_clubs[team] = current_clubs.get(team, 0) + 1
            
        # --- PHASE 2: FALLBACK (Fill gaps with Cheapest Valid Players) ---
        # If we ran out of budget or valid high-value players, fill the rest with cheap enablers
        if len(selected_ids) < self.constraints.total_players:
            logger.info("Greedy pass incomplete. Running fallback fill...")
            
            # Sort by COST (Ascending) -> Cheapest players first
            remaining_df = df_opt[~df_opt['id'].isin(selected_ids)].sort_values('cost', ascending=True)
            
            for _, player in remaining_df.iterrows():
                if len(selected_ids) == self.constraints.total_players: break
                
                pid = player['id']
                pos = player['position']
                team = player['team']
                cost = player['cost']
                
                # STRICT CONSTRAINT CHECKS FOR FALLBACK
                if current_cost + cost > self.constraints.budget: continue
                if current_pos.get(pos, 0) >= self.constraints.positions.get(pos, 0): continue
                if current_clubs.get(team, 0) >= self.constraints.max_per_club: continue
                
                # Select Player
                selected_ids.append(pid)
                current_cost += cost
                current_pos[pos] = current_pos.get(pos, 0) + 1
                current_clubs[team] = current_clubs.get(team, 0) + 1

        # Calculate Final Results
        squad_df = df[df['id'].isin(selected_ids)]
        total_points = squad_df['id'].map(predicted_points).sum()
        is_valid, violations = self.validate_solution(selected_ids, df)
        
        return SquadSolution(
            player_ids=selected_ids,
            total_cost=current_cost,
            total_points=total_points,
            position_distribution=current_pos,
            club_distribution=current_clubs,
            is_valid=is_valid,
            violations=violations
        )

    def get_squad_details(self, solution: SquadSolution, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to return a readable DataFrame of the selected squad.
        """
        cols = ['id', 'name', 'team', 'position', 'cost', 'points']
        # Ensure columns exist
        available_cols = [c for c in cols if c in df.columns]
        
        res = df[df['id'].isin(solution.player_ids)][available_cols].copy()
        
        # Add predicted points if available in the solution object context (not stored directly, but good practice)
        return res

if __name__ == "__main__":
    # Simple Test
    from ml.data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    df = ingestion.generate_mock_data(n_players=500)
    
    # Mock Predictions
    preds = {pid: np.random.randint(50, 200) for pid in df['id']}
    
    optimizer = CSPOptimizer()
    sol = optimizer.optimize_squad(df, preds)
    
    print(f"Valid: {sol.is_valid}")
    print(f"Cost: {sol.total_cost}")
    print(f"Points: {sol.total_points}")
    print(f"Violations: {sol.violations}")