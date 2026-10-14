from geo_inference.geo_inference import GeoInference

def main() -> None:
    CHECKPOINT_PATH = "/export/sata01/wspace/test_dir/multi/demo/hf/sam3.pt"
    BPE_PATH = "/export/sata01/wspace/test_dir/multi/demo/hf/tokenizer/bpe_simple_vocab_16e6.txt.gz"
    geo_inference = GeoInference(
        model="/export/sata01/wspace/test_dir/multi/demo/pt2/dofav2_67_0.418.pt2",
        work_dir="/export/sata01/wspace/test_dir/multi/demo/prediction",
        device="gpu",
        post_inference = True,
        sam_checkpoint_path=CHECKPOINT_PATH,
        sam_bpe_path=BPE_PATH,
    )

    image_path = "/export/sata01/wspace/test_dir/multi/demo/image/ns2_small_rgbn.tif"
    geo_inference(
        inference_input=image_path,
        sensor_name="worldview-2-rgbn",
        patch_size=512,
    )

if __name__ == "__main__":
    main()
