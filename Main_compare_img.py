import cv2
import numpy as np


def find_filled_answers(image_with_answers, image_without_answers):
    gray_with_answers = cv2.cvtColor(image_with_answers, cv2.COLOR_BGR2GRAY)
    gray_without_answers = cv2.cvtColor(image_without_answers, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_with_answers, gray_without_answers)
    _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100
    filled_positions = [(x, y) for contour in contours if cv2.contourArea(contour) > min_contour_area for x, y, _, _ in
                        [cv2.boundingRect(contour)]]

    return filled_positions, contours


def main():
    filled_answer_sheet_path = "F:\Shubham\Fill in the blanks answe1.jpeg"
    empty_question_paper_path = "F:\Shubham\Fill in the blanks empty sheet.jpg"

    filled_answer_sheet = cv2.imread(filled_answer_sheet_path)
    empty_question_paper = cv2.imread(empty_question_paper_path)

    if filled_answer_sheet is None or empty_question_paper is None:
        print("Error: Unable to load images.")
        return

    filled_positions, contours = find_filled_answers(filled_answer_sheet, empty_question_paper)

    print("The filled answers are:")
    for i, (x, y) in enumerate(filled_positions, start=1):
        print(f"ans{i} - Position: ({x}, {y})")

    # Draw contours on the filled_answer_sheet and display the image
    image_with_contours = filled_answer_sheet.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
