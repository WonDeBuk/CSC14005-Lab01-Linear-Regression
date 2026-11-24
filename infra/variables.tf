variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "co2-studio"
}

variable "region" {
  description = "Region of the S3 bucket (CloudFront itself is global)."
  type        = string
  default     = "ap-southeast-1"
}

variable "site_dir" {
  description = "Built web app to upload (run `npm run build` in web/ first)."
  type        = string
  default     = "../web/dist"
}

variable "price_class" {
  description = "CloudFront edge locations. PriceClass_200 includes Asia."
  type        = string
  default     = "PriceClass_200"
}
