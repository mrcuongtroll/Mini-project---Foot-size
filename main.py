import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import foot


def main():
    st.title("SHOE SIZE ESTIMATOR")
    st.write("You can upload a photo of your foot on an A4 paper and we will calculate your shoe size.")
    st.markdown("#### IMPORTANT: Your image MUST satisfy the following criteria:\n"
                "- The image must be in top-down view.\n"
                "- All 4 corners of the paper must be visible in the images.\n"
                "- The foot must be straight in the direction of the paper.\n"
                "- Ideally your heel should be placed roughly 2 cm from the edge of the paper."
                " Heel touches on edge of the paper are also acceptable.\n"
                "- The floor must have no pattern and the color should not be to bright."
                " We want to avoid the case where the paper is indistinguishable from the floor.\n"
                "- Please take the photo in a well-lit place. The less shadows in the image, the better.\n"
                "- We will show you all intermediate processing steps."
                " If you find any error in these steps, please retake your photo.\n",
                True)
    image_file = st.file_uploader("Upload image")
    if image_file:
        st.header("Uploaded image:")
        st.image(image_file)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        # Load image
        # img = cv2.imread('./images/23809.jpeg')
        img = cv2.imdecode(file_bytes, 1)
        img = foot.laplacian_sharper(img)
        # img = cv2.imread('./barefeet1.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # fig0, ax0 = plt.subplots()
        # ax0.imshow(img)
        # plt.show()
        # st.pyplot(fig0)
        # img = cv2.GaussianBlur(img, (7, 7), 0)
        measure_method = st.radio("Measure based on:", ('Foot length', 'Foot width'), index=0)
        if st.button("Calculate shoe size"):
            col1, col2, col3, col4 = st.columns(4)
            # --- Phase 1: find paper bounding box ---
            # Segment image. We use only 2 clusters to separate the foot from the paper
            segmented_img = foot.segment(img, 2)
            # st.header("Semantic segmentation:")
            col1.subheader("Semantic segmentation:")
            fig1, ax1 = plt.subplots()
            ax1.imshow(segmented_img, cmap='gray')
            plt.show()
            # st.pyplot(fig1)
            col1.pyplot(fig1)
            # Detect edge from segmented image
            # segmented_img = cv2.GaussianBlur(segmented_img, (5, 5), 0)
            edged_img = foot.detect_edge(segmented_img)
            # st.header("Edge:")
            col2.subheader("Edge detection:")
            fig2, ax2 = plt.subplots()
            ax2.imshow(edged_img, cmap='gray')
            plt.show()
            # st.pyplot(fig2)
            col2.pyplot(fig2)
            # Get contour and bounding boxes
            orig = img.copy()
            paper_bbox, paper_box_points = foot.get_bounding_box(edged_img, orig)
            pr_width = int(paper_bbox[1][0])
            pr_height = int(paper_bbox[1][1])
            src_pts = paper_box_points.astype("float32")
            dst_pts = np.array([[0, pr_height - 1],
                                [0, 0],
                                [pr_width - 1, 0],
                                [pr_width - 1, pr_height - 1]], dtype="float32")
            cropped_img = cv2.warpPerspective(orig, cv2.getPerspectiveTransform(src_pts, dst_pts), (pr_width, pr_height))
            # st.header("Paper detection:")
            col3.subheader("Paper detection:")
            fig3, ax3 = plt.subplots()
            ax3.imshow(orig)
            plt.show()
            # st.pyplot(fig3)
            col3.pyplot(fig3)

            # --- Phase 2: find foot bounding box ---
            # Find bounding box for the foot
            crop_rate = 0.025
            if pr_height < pr_width:
                pr_height, pr_width = pr_width, pr_height
                cropped_img = cropped_img[int(pr_width * crop_rate):int(pr_width * (1 - crop_rate)),
                              int(pr_height * crop_rate):int(pr_height * (1 - crop_rate)), :]
            else:
                cropped_img = cropped_img[int(pr_height * crop_rate):int(pr_height * (1 - crop_rate)),
                              int(pr_width * crop_rate):int(pr_width * (1 - crop_rate)), :]
            segmented_cropped_img = foot.segment(cropped_img, 2)
            edged_cropped_img = foot.detect_edge(segmented_cropped_img)
            foot_bbox, foot_box_points = foot.get_bounding_box(edged_cropped_img, cropped_img)
            fr_width = int(foot_bbox[1][0])
            fr_height = int(foot_bbox[1][1])
            if fr_height < fr_width:
                fr_height, fr_width = fr_width, fr_height
            # st.header("Foot detection:")
            col4.subheader("Foot detection:")
            fig4, ax4 = plt.subplots()
            ax4.imshow(cropped_img)
            plt.show()
            print(f'Paper box: {pr_width} - {pr_height}')
            print(f'Foot box: {fr_width} - {fr_height}')
            print(
                f'Width: {(fr_width / pr_width * 210 / 10):.2f} cm.\t Length: {(fr_height / pr_height * 297 / 10):.2f} cm.')
            # st.pyplot(fig4)
            col4.pyplot(fig4)

            # --- Display results
            foot_width = fr_width / pr_width * 210 / 10
            foot_length = fr_height / pr_height * 297 / 10
            shoe_size = None
            if measure_method == 'Foot length':
                shoe_size = foot.convert_to_shoe_size(foot_length, measure='length')
                if shoe_size is None:
                    shoe_size = foot.convert_to_shoe_size(foot_width, measure='width')
            elif measure_method == 'Foot width':
                shoe_size = foot.convert_to_shoe_size(foot_width, measure='width')
                if shoe_size is None:
                    shoe_size = foot.convert_to_shoe_size(foot_length, measure='length')
            if shoe_size is None:
                shoe_size = {"US": 6, "UK": 5, 'VN': 39}
            st.header("Approximated shoe size:")
            st.markdown(f'Foot width: {foot_width:.2f} cm.<br>'
                        f'Foot length: {foot_length:.2f} cm.<br>'
                        f'Shoe size: {shoe_size["US"]} (US)&emsp;{shoe_size["UK"]} (UK)&emsp;{shoe_size["VN"]} (VN)',
                        True)

            # --- Option to choose another image ---
            if st.button("Choose another image"):
                main()


if __name__ == '__main__':
    main()
