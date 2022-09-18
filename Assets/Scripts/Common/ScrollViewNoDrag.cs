using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;


public class ScrollViewNoDrag : ScrollRect
{
    public override void OnDrag(PointerEventData eventData)
    {
        return;
    }
    public override void OnBeginDrag(PointerEventData eventData)
    {
        return;
    }
    public override void OnEndDrag(PointerEventData eventData)
    {
        return;
    }
}
