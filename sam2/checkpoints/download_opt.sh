#!/bin/bash

mkdir opts
cd opts
wget -O sam2_opt_models.tar.gz "https://shanhai-ai.bd.bcebos.com/v1/FastProcess/models/common/sam2_opt/sam2_opt_models.tar.gz?authorization=bce-auth-v1%2FALTAKKDx9mToS4enALDWmlY4yl%2F2025-07-30T09%3A06%3A21Z%2F-1%2Fhost%2F5cee6fcb917c4e86043d47f2018928a2027fc5d7beb5f61a6dc73848e4159def"
tar -xavf sam2_opt_models.tar.gz
cd ..