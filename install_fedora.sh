cat > install_fedora.sh << 'EOF'
#!/bin/bash
# ============================================================
# Скрипт установки окружения для ozone-flow-digital-twin
# ОС: Fedora 41/42/43
# Автор: команда хакатона
# ============================================================

set -e  # Остановить при ошибке (от англ. *exit on error*)

echo "🚀 Начинаем установку окружения..."

# ------------------------------------------------------------
# 1. Системные пакеты (через dnf)
# ------------------------------------------------------------
echo "📦 Устанавливаем системные пакеты..."
sudo dnf update -y
sudo dnf install -y \
    git \
    python3.12 \
    python3.12-devel \
    python3-pip \
    ffmpeg \
    flatpak

# ------------------------------------------------------------
# 2. Docker (официальный репозиторий)
# ------------------------------------------------------------
echo "🐳 Устанавливаем Docker..."
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Запускаем Docker и добавляем пользователя в группу
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
echo "⚠️  ВАЖНО: перезайдите в терминал или выполните 'newgrp docker', чтобы Docker работал без sudo"

# ------------------------------------------------------------
# 3. Node.js (для 3D-визуализации через Three.js)
# ------------------------------------------------------------
echo "🌐 Устанавливаем Node.js..."
curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
sudo dnf install -y nodejs

# ------------------------------------------------------------
# 4. Blender через Flatpak (3D-моделирование)
# ------------------------------------------------------------
echo "🎨 Устанавливаем Blender..."
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install -y flathub org.blender.Blender

# ------------------------------------------------------------
# 5. Python виртуальное окружение
# ------------------------------------------------------------
echo "🐍 Создаём Python окружение..."
python3.12 -m venv .venv
source .venv/bin/activate

# ------------------------------------------------------------
# 6. Python библиотеки
# ------------------------------------------------------------
echo "📚 Устанавливаем Python библиотеки..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------
# 7. Проверка установки
# ------------------------------------------------------------
echo ""
echo "✅ Проверка установленных компонентов:"
echo "   Git: $(git --version)"
echo "   Python: $(python3.12 --version)"
echo "   Docker: $(docker --version)"
echo "   Node.js: $(node --version)"
echo "   Blender: $(flatpak run org.blender.Blender --version 2>/dev/null || echo 'установлен через flatpak')"
echo ""
echo "🎉 Установка завершена!"
echo "   Для активации окружения выполните: source .venv/bin/activate"
EOF

# Делаем скрипт исполняемым (чтобы его можно было запустить)
chmod +x install_fedora.sh