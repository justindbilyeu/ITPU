cat > itpu/__init__.py << 'EOF'
# SPDX-License-Identifier: Apache-2.0
"""
Information-Theoretic Processing Unit (ITPU)
Software SDK for entropy, mutual information, and k-NN statistics.
"""

__version__ = "0.1.0"

from .sdk import ITPU
from .utils.windowed import windowed_mi

__all__ = ["ITPU", "windowed_mi"]
EOF
