#!/usr/bin/env bash
# Build the web app and publish it to S3 + CloudFront.
#
#   infra/deploy.sh              # deploy (creates the stack on first run)
#   infra/deploy.sh --destroy    # empty the bucket and delete everything
#
# Uses the current AWS CLI credentials (AWS_PROFILE / AWS_REGION are honoured).
set -euo pipefail

STACK="${STACK:-co2-studio}"
REGION="${AWS_REGION:-ap-southeast-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

output() {
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

if [[ "${1:-}" == "--destroy" ]]; then
  aws s3 rm "s3://$(output BucketName)" --recursive --region "$REGION"
  aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
  echo "Deleted stack $STACK"
  exit 0
fi

echo "==> Exporting model coefficients"
PY="${PYTHON:-$ROOT/.venv/bin/python}"; [[ -x "$PY" ]] || PY=python3
(cd "$ROOT" && "$PY" -m linreg.export_web)

echo "==> Building web app"
(cd "$ROOT/web" && npm ci --silent && npm run build)

echo "==> Deploying stack $STACK in $REGION (first run takes ~5 minutes for CloudFront)"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file "$ROOT/infra/web-hosting.yaml" --no-fail-on-empty-changeset \
  --tags Project=CSC14005-Lab01

BUCKET="$(output BucketName)"
DIST="$(output DistributionId)"

echo "==> Uploading to s3://$BUCKET"
# Hashed assets never change: cache for a year. Entry files: always revalidate.
aws s3 sync "$ROOT/web/dist/assets" "s3://$BUCKET/assets" --region "$REGION" --delete \
  --cache-control "public,max-age=31536000,immutable"
aws s3 sync "$ROOT/web/dist" "s3://$BUCKET" --region "$REGION" --delete --exclude "assets/*" \
  --cache-control "public,max-age=0,must-revalidate"

echo "==> Invalidating CloudFront cache"
aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/*" --query Invalidation.Id --output text

echo "==> Live at $(output SiteUrl)"
