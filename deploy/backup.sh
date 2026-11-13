#!/usr/bin/env bash
# Copies the live database out of the container into ~/backups with a date.
# Run by hand, or daily via cron:  0 23 * * * /home/ubuntu/attendance_system/deploy/backup.sh
set -euo pipefail
mkdir -p "$HOME/backups"
STAMP=$(date +%Y-%m-%d_%H%M)
docker exec attendance python -c "import sqlite3; s=sqlite3.connect('/data/attendance.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d); d.close()"
docker cp attendance:/data/backup.db "$HOME/backups/attendance_$STAMP.db"
docker exec attendance rm -f /data/backup.db
ls -1t "$HOME/backups"/attendance_*.db | tail -n +31 | xargs -r rm -f   # keep last 30
echo "Backup saved: $HOME/backups/attendance_$STAMP.db"
