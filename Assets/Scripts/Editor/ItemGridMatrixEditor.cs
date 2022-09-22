using UnityEditor.UIElements;
using UnityEngine.UIElements;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(ItemGridMatrix))]
public class ItemGridMatrixEditor : Editor
{
    public VisualTreeAsset uiAsset;
    VisualElement gridParent;
    bool needsGridUpdate = true;
    public override VisualElement CreateInspectorGUI()
    {
        ItemGridMatrix matrix = (ItemGridMatrix) target;
        VisualElement myInspector = new VisualElement();

        uiAsset.CloneTree(myInspector);

        var sizeXField = myInspector.Q("sizeXField");
        sizeXField.RegisterCallback<ChangeEvent<int>>(SizeChangedEvent);

        var sizeYField = myInspector.Q("sizeYField");
        sizeYField.RegisterCallback<ChangeEvent<int>>(SizeChangedEvent);

        var gridElement = myInspector.Q("gridVisualElement");
        gridParent = gridElement;
        if (matrix.sizeX <= 0 || matrix.sizeY <= 0) return myInspector;

        if(needsGridUpdate) UpdateGrid(gridParent);

        var debugButton = myInspector.Q("dbgPrintButton");
        ((Button)debugButton).clicked += matrix.Print;

        return myInspector;
    }

    void UpdateGrid(VisualElement parent)
    {
        parent.Clear();

        var matrix = (ItemGridMatrix)target;
        var serMatrix = new SerializedObject(target);
        var grid = serMatrix.FindProperty("occupiedMatrix");
        for (int i = 0; i < matrix.sizeY; i++)
        {
            var _visualElement = new VisualElement();
            _visualElement.name = "subGridElement";
            for (int j = 0; j < matrix.sizeX; j++)
            {
                var _property = grid.GetArrayElementAtIndex(i * matrix.sizeX + j);
                var _toggle = new Toggle("");
                _toggle.BindProperty(_property);
                _visualElement.Add(_toggle);
            }
            parent.Add(_visualElement);
        }
        needsGridUpdate = false;
    }

    void SizeChangedEvent(ChangeEvent<int> sizeX)
    {
        ((ItemGridMatrix)target).UpdateMatrix();
        needsGridUpdate = true;
        UpdateGrid(gridParent);
    }
}
