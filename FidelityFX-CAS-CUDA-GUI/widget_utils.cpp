#include "widget_utils.hpp"

void WidgetUtils::scalePixmap(QPixmap &pixmap, const QSize targetImageSize)
{
    // Only scale down if the image is larger than the target size
    if (pixmap.size().width() > targetImageSize.width() || pixmap.size().height() > targetImageSize.height())
        pixmap = pixmap.scaled(targetImageSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}
