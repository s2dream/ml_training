import torch
import torch.nn as nn
import torch.nn.functional as FD
from typing import Optional

# -------------------------
# 유틸: Sinusoidal Positional Encoding
# -------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1, T, D]

# -------------------------
# 메인 모델
# -------------------------
class MetroRegressor(nn.Module):
    def __init__(
        self,
        TOTAL_SIZE_METRO_ITEM: int,
        TOTAL_SIZE_METRO_STEP: int,   # 요구사항 4에 맞춰 이름 유지
        dim: int = 512,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        combine_mode: str = "concat_project",  # "concat_project" 또는 "sum"
        max_main_seq_len: int = 4096,          # positional encoding 길이
    ):
        super().__init__()
        self.dim = dim
        self.combine_mode = combine_mode

        # 1) metro_item_set_id 임베딩 (padding_idx=0 가정)
        self.item_emb = nn.Embedding(
            num_embeddings=TOTAL_SIZE_METRO_ITEM,
            embedding_dim=dim,
            padding_idx=0
        )

        # 2) metro_item_value → 512로 임베딩 (작은 MLP)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

        # 3) item-ID 임베딩과 value 임베딩 결합
        if combine_mode == "concat_project":
            self.item_combine = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.LayerNorm(dim)
            )
        elif combine_mode == "sum":
            self.item_combine = nn.Identity()
        else:
            raise ValueError("combine_mode must be 'concat_project' or 'sum'.")

        # 4) metro_set_id 임베딩 (padding_idx=0 가정)
        self.set_emb = nn.Embedding(
            num_embeddings=TOTAL_SIZE_METRO_STEP,
            embedding_dim=dim,
            padding_idx=0
        )

        # 7) Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_enc = SinusoidalPE(d_model=dim, max_len=max_main_seq_len)

        # 7) 3-레이어 MLP 회귀 헤드
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )

    def forward(
        self,
        x_metro_item_set_id: torch.LongTensor,   # [B, T, S, I]
        x_metro_item_value: torch.FloatTensor,   # [B, T, S, I]
        x_metro_set_id: torch.LongTensor,        # [B, T, S]
        padding_mask_main_steps: torch.BoolTensor,  # [B, T]  True=pad
    ):
        """
        B: batch_size
        T: main_seq_max_len
        S: metro_set_max_len
        I: metro_item_set_max_len
        dim: 512
        출력: y_pred [B, 1]
        """

        B, T, S, I = x_metro_item_set_id.shape

        # ----------- (1) item ID 임베딩 -----------
        # [B, T, S, I, D]
        item_id_emb = self.item_emb(x_metro_item_set_id)

        # ----------- (2) value 임베딩 -----------
        # value는 실수 → [B, T, S, I, 1]로 확장 후 MLP
        value_in = x_metro_item_value.unsqueeze(-1)
        value_emb = self.value_mlp(value_in)  # [B, T, S, I, D]

        # padding용 마스크 (item 차원): id==0 이면 padding으로 간주
        item_valid_mask = (x_metro_item_set_id != 0).unsqueeze(-1)  # [B, T, S, I, 1]

        # ----------- (3) item ID & value 결합 + item aggregation(sum over I) -----------
        if self.combine_mode == "concat_project":
            combined = torch.cat([item_id_emb, value_emb], dim=-1)       # [B, T, S, I, 2D]
            combined = self.item_combine(combined)                       # [B, T, S, I, D]
        else:  # 'sum'
            combined = item_id_emb + value_emb                           # [B, T, S, I, D]

        # padding 위치는 0으로
        combined = combined * item_valid_mask

        # I 차원 합 → [B, T, S, D]
        items_agg = combined.sum(dim=3)

        # ----------- (4) metro_set_id 임베딩 -----------
        # [B, T, S, D]
        set_emb = self.set_emb(x_metro_set_id)

        # set 레벨 마스크: id==0 → padding
        set_valid_mask = (x_metro_set_id != 0).unsqueeze(-1)  # [B, T, S, 1]

        # ----------- (5) (3)과 (4) 합 -----------
        sets_enriched = (items_agg + set_emb) * set_valid_mask  # [B, T, S, D]

        # ----------- (6) S 차원 aggregation(sum) → [B, T, D] -----------
        token_repr = sets_enriched.sum(dim=2)  # [B, T, D]

        # ----------- (7) Transformer + 3-layer MLP로 y 예측 -----------
        # Positional Encoding
        token_repr = self.pos_enc(token_repr)  # [B, T, D]

        # Transformer 인코더
        # src_key_padding_mask: [B, T], True=pad
        encoded = self.encoder(token_repr, src_key_padding_mask=padding_mask_main_steps)  # [B, T, D]

        # 회귀 타깃 y가 [B, 1] 하나이므로, "특정 토큰 하나"를 연결해 MLP 태움.
        # 일반적으로 sequence의 첫 토큰(index 0)을 대표 토큰으로 사용(CLS-like).
        rep_token = encoded[:, 0, :]  # [B, D]
        y_pred = self.head(rep_token)  # [B, 1]
        return y_pred

# -------------------------
# 간단한 사용 예시 (더미 텐서)
# -------------------------
if __name__ == "__main__":
    B, T, S, I = 2, 5, 7, 11
    TOTAL_SIZE_METRO_ITEM = 50000
    TOTAL_SIZE_METRO_STEP = 10000
    dim = 512

    model = MetroRegressor(
        TOTAL_SIZE_METRO_ITEM=TOTAL_SIZE_METRO_ITEM,
        TOTAL_SIZE_METRO_STEP=TOTAL_SIZE_METRO_STEP,
        dim=dim,
        nhead=8,
        num_transformer_layers=2,
        combine_mode="concat_project",   # "sum"으로 바꿔 실험해보세요
        max_main_seq_len=8192
    )

    # 더미 입력 (0은 padding 가정)
    x_metro_item_set_id = torch.randint(0, TOTAL_SIZE_METRO_ITEM, (B, T, S, I))
    x_metro_item_value = torch.randn(B, T, S, I)
    x_metro_set_id = torch.randint(0, TOTAL_SIZE_METRO_STEP, (B, T, S))
    padding_mask_main_steps = torch.zeros(B, T, dtype=torch.bool)  # 전부 유효(패딩 없음) 예시

    y_pred = model(
        x_metro_item_set_id=x_metro_item_set_id,
        x_metro_item_value=x_metro_item_value,
        x_metro_set_id=x_metro_set_id,
        padding_mask_main_steps=padding_mask_main_steps
    )
    print(y_pred.shape)  # [B, 1]
