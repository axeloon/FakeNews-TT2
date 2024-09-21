"""Crear tabla noticias_redes_sociales

Revision ID: 05a047ab6769
Revises: 
Create Date: 2024-09-21 17:08:07.991390

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '05a047ab6769'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('noticias_redes_sociales',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source', sa.String(length=50), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('publication_date', sa.Date(), nullable=True),
    sa.Column('author', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_noticias_redes_sociales_id'), 'noticias_redes_sociales', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_noticias_redes_sociales_id'), table_name='noticias_redes_sociales')
    op.drop_table('noticias_redes_sociales')